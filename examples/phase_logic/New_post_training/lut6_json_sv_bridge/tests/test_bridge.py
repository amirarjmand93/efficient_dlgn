#!/usr/bin/env python3
"""Compiler tests: exact original-regression hash, synthetic widths/shape/depths,
full and compact ROM decode, independent direct-vs-compiled evaluation.
Run python -m unittest discover -s tests -v from bridge root.
"""
from __future__ import annotations
import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lut6_json_bridge.compiler import (IRValidationError,validated_model,compile_instructions,
    encode_program,decode_program,eval_reference,eval_compiled,build)


def model(channels=2, hw=4, convs=1, denses=(12,8),classes=2,seed=99):
    """A legal linear LUT6 IR; convs can be 0..N, dense widths arbitrary."""
    rng=random.Random(seed)
    layers=[]
    prior_shape=[channels,hw,hw] if convs else [channels*hw*hw]
    prior_name='encoded_input'
    for k in range(convs):
        cin,h,w=prior_shape
        cout=3+k
        kh,kw=2,2
        out_shape=[cout,h-1,w-1]
        pins=[[rng.randrange(cin*kh*kw) for _ in range(6)] for _ in range(cout)]
        name=f'conv{k+1}'
        layers.append(dict(name=name,op_type='LUT6Conv2D',input=prior_name,
            input_shape=prior_shape,output_shape=out_shape,kernel_size=[kh,kw],
            stride=[1,1],padding=[0,0],padding_value=0,
            local_patch_flattening=dict(order='channel,row,column'),
            connections=pins,init_hex=[f'{rng.getrandbits(64):016X}' for _ in range(cout)],
            pin0_is_address_lsb=True,weight_sharing=dict(parameter_index='output_channel')))
        prior_name=name
        prior_shape=out_shape
    for k,width in enumerate(denses):
        flat_len=1
        for n in prior_shape:flat_len*=n
        name=f'dense{k+1}'
        pins=[[rng.randrange(flat_len) for _ in range(6)] for _ in range(width)]
        layers.append(dict(name=name,op_type='LUT6Dense',input=prior_name,
            input_shape=[flat_len],output_shape=[width],connections=pins,
            init_hex=[f'{rng.getrandbits(64):016X}' for _ in range(width)],pin0_is_address_lsb=True))
        prior_name=name
        prior_shape=[width]
    out_count=prior_shape[0]
    class_groups=[]
    # Keep intervals nonoverlapping, class groups may be uneven.
    for k in range(classes):
        lo=(k*out_count)//classes
        hi=((k+1)*out_count)//classes
        assert hi>lo
        class_groups.append(dict(class_index=k,node_start=lo,node_stop_exclusive=hi))
    layers.append(dict(name='group_sum',op_type='BooleanGroupSum',input=prior_name,
        input_shape=prior_shape,output_shape=[classes],classes=classes,class_groups=class_groups))
    return dict(format='hls4ml_lut6_ir_v1',input=dict(name='encoded_input',dtype='boolean',
        shape=[channels,hw,hw] if convs else [channels*hw*hw],
        layout='CHW' if convs else 'flat',flattening=dict(order='channel,row,column')),
        layers=layers)


class CompilerTests(unittest.TestCase):
    def test_shapes_and_depths(self):
        for convs,denses,channels,hw in [
            (0,(9,),1,2),(0,(7,9,11),2,2),(1,(10,),3,5),
            (1,(13,9),12,4),(2,(7,8,6),2,5),(3,(12,),1,5)
        ]:
            with self.subTest(convs=convs,denses=denses,channels=channels):
                ir=model(channels,hw,convs,denses)
                spec=validated_model(ir)
                instr=compile_instructions(spec)
                self.assertEqual(len(instr),spec['n_program'])
                rng=random.Random(4)
                bits=[rng.getrandbits(1) for _ in range(spec['n_input'])]
                for encoding in ('full','split'):
                    words,tables,width,id_w=encode_program(instr,spec['addr_w'],encoding)
                    decoded=decode_program(spec,words,tables,encoding,id_w)
                    self.assertEqual(decoded,instr)
                    self.assertEqual(eval_reference(ir,spec,bits),eval_compiled(spec,decoded,bits))

    def test_all_inputs_exhaustive(self):
        # Single dense layer, 4 inputs => exhaustive all 16 possible inputs.
        ir=model(channels=1,hw=2,convs=0,denses=(8,))
        spec=validated_model(ir)
        instr=compile_instructions(spec)
        for k in range(16):
            bits=[(k>>i)&1 for i in range(4)]
            self.assertEqual(eval_reference(ir,spec,bits),eval_compiled(spec,instr,bits))

    def test_padding_and_per_output_conv(self):
        ir=model(channels=2,hw=4,convs=1,denses=(8,))
        layer=ir['layers'][0]
        layer['padding']=[1,1]
        layer['output_shape']=[3,5,5]
        # Make per-output rather than channel-shared LUT parameters
        layer['connections']=[list(layer['connections'][c]) for c in range(3) for _ in range(25)]
        layer['init_hex']=[layer['init_hex'][c] for c in range(3) for _ in range(25)]
        ir['layers'][1]['input_shape']=[75]
        spec=validated_model(ir)
        instr=compile_instructions(spec)
        pad=(1<<spec['addr_w'])-1
        self.assertTrue(any(pad in pins for _,pins in instr))
        bits=[(i%7)==3 for i in range(spec['n_input'])]
        self.assertEqual(eval_reference(ir,spec,bits),eval_compiled(spec,instr,bits))

    def test_large_input_12ch_and_codegen(self):
        ir=model(channels=12,hw=32,convs=1,denses=(9,8))
        # Initial conv output is 3*31*31=2883, source input 12288.
        spec=validated_model(ir)
        self.assertEqual(spec['n_input'],12288)
        self.assertEqual(spec['addr_w'],14)
        with tempfile.TemporaryDirectory() as folder:
            for encoding in ('full','split'):
                out=Path(folder)/encoding
                manifest=build_json(ir,out,encoding)
                self.assertEqual(manifest['input_bits'],12288)
                rtl=(out/'rtl'/'lut6_sequential.sv').read_text()
                self.assertIn('BANK0_DEPTH=',rtl)
                self.assertTrue((out/'sim'/'tb_lut6_sequential.sv').exists())

    def test_reject_skip(self):
        ir=model(convs=1,denses=(9,8))
        ir['layers'][2]['input']='encoded_input'
        with self.assertRaises(IRValidationError):validated_model(ir)

    def test_original_sha_and_counts_if_present(self):
        # Locate a copy only in our local development checkout; portable test skips.
        source=ROOT/'examples'/'reference_original_ir.json'
        if not source.exists():self.skipTest('Optional original IR not bundled')
        ir=json.loads(source.read_text())
        spec=validated_model(ir)
        inst=compile_instructions(spec)
        words,_,width,_=encode_program(inst,spec['addr_w'],'full')
        import hashlib
        digest=hashlib.sha256(('\n'.join(words)+'\n').encode()).hexdigest()
        self.assertEqual(digest,'3f1d3da2b0c6c5005721a5c3d99aabad23fd9c76c60d49450910f0cb32bf3401')
        bits=[0]*spec['n_input']
        self.assertEqual(eval_reference(ir,spec,bits),[438,293,444,471,417,350,422,362,390,243])


def build_json(ir,out,encoding):
    f=out.parent/'src.json'
    f.write_text(json.dumps(ir))
    return build(f,out,encoding,n_random=0)


if __name__=='__main__':unittest.main()
