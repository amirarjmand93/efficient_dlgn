#ifndef NNET_LUT6_H_
#define NNET_LUT6_H_

#include "nnet_common.h"

namespace nnet {

struct lut6_conv2d_config {};
struct lut6_dense_config {};
struct boolean_group_sum_config {};

// A bank is one 64-bit word. Complete partitioning in the generated top
// separates words, so no intermediate becomes an 8,000+ bit HLS vector.
template<class input_T>
ap_uint<1> lut6_read(const input_T *input, unsigned index) {
    #pragma HLS INLINE
    return input[index];
}

inline ap_uint<1> lut6_read(const ap_uint<64> *input, unsigned index) {
    #pragma HLS INLINE
    return input[index / 64][index % 64];
}

template<unsigned FILTER, class input_T, typename CONFIG_T>
void lut6_conv_filter(const input_T *input,
                      ap_uint<64> output[CONFIG_T::banks_per_filter]) {
    #pragma HLS INLINE off
    static_assert(CONFIG_T::bank_width == 64, "LUT6 banks must be 64 bits");
    static_assert(CONFIG_T::out_height * CONFIG_T::out_width ==
                  CONFIG_T::banks_per_filter * 64,
                  "Convolution filter output must fill whole banks");
    for (unsigned bank = 0; bank < CONFIG_T::banks_per_filter; bank++) {
        #pragma HLS UNROLL
        ap_uint<64> word = 0;
        for (unsigned offset = 0; offset < 64; offset++) {
            #pragma HLS UNROLL
            const unsigned location = bank * 64 + offset;
            const unsigned out_row = location / CONFIG_T::out_width;
            const unsigned out_col = location % CONFIG_T::out_width;
            ap_uint<6> address = 0;
            for (unsigned pin = 0; pin < 6; pin++) {
                #pragma HLS UNROLL
                const unsigned connection = CONFIG_T::connections[FILTER][pin];
                const unsigned channel =
                    connection / (CONFIG_T::kernel_height * CONFIG_T::kernel_width);
                const unsigned patch_index =
                    connection % (CONFIG_T::kernel_height * CONFIG_T::kernel_width);
                const int in_row = static_cast<int>(out_row * CONFIG_T::stride_height)
                                 + static_cast<int>(patch_index / CONFIG_T::kernel_width)
                                 - static_cast<int>(CONFIG_T::pad_height);
                const int in_col = static_cast<int>(out_col * CONFIG_T::stride_width)
                                 + static_cast<int>(patch_index % CONFIG_T::kernel_width)
                                 - static_cast<int>(CONFIG_T::pad_width);
                ap_uint<1> bit = CONFIG_T::padding_value;
                if (in_row >= 0 && in_row < static_cast<int>(CONFIG_T::in_height)
                        && in_col >= 0 && in_col < static_cast<int>(CONFIG_T::in_width)) {
                    const unsigned input_index =
                        (channel * CONFIG_T::in_height + in_row) * CONFIG_T::in_width + in_col;
                    bit = lut6_read(input, input_index);
                }
                address[pin] = bit;
            }
            word[offset] = CONFIG_T::truth_tables[FILTER][address];
        }
        output[bank] = word;
    }
}

template<unsigned FILTER, class input_T, typename CONFIG_T,
         bool DONE = (FILTER == CONFIG_T::out_channels)>
struct lut6_conv_filters;

template<unsigned FILTER, class input_T, typename CONFIG_T>
struct lut6_conv_filters<FILTER, input_T, CONFIG_T, false> {
    static void run(const input_T *input, ap_uint<64> *output) {
        #pragma HLS INLINE
        lut6_conv_filter<FILTER, input_T, CONFIG_T>(
            input, output + FILTER * CONFIG_T::banks_per_filter
        );
        lut6_conv_filters<FILTER + 1, input_T, CONFIG_T>::run(input, output);
    }
};

template<unsigned FILTER, class input_T, typename CONFIG_T>
struct lut6_conv_filters<FILTER, input_T, CONFIG_T, true> {
    static void run(const input_T *, ap_uint<64> *) {}
};

template<class data_T, class result_T, typename CONFIG_T>
void lut6_conv2d_flat(data_T input[CONFIG_T::n_in],
                      ap_uint<64> output[CONFIG_T::output_banks]) {
    lut6_conv_filters<0, data_T, CONFIG_T>::run(input, output);
}

template<class data_T, class result_T, typename CONFIG_T>
void lut6_conv2d_packed(ap_uint<64> input[(CONFIG_T::n_in + 63) / 64],
                        ap_uint<64> output[CONFIG_T::output_banks]) {
    lut6_conv_filters<0, ap_uint<64>, CONFIG_T>::run(input, output);
}

template<unsigned TILE, typename CONFIG_T>
void lut6_dense_tile(const ap_uint<64> *input, ap_uint<64> output[2]) {
    #pragma HLS INLINE off
    static_assert(CONFIG_T::bank_width == 64 && CONFIG_T::tile_size == 128,
                  "Dense tiles require two 64-bit output banks");
    ap_uint<64> first_word = 0;
    ap_uint<64> second_word = 0;
    for (unsigned offset = 0; offset < CONFIG_T::tile_size; offset++) {
        #pragma HLS UNROLL
        const unsigned node = TILE * CONFIG_T::tile_size + offset;
        if (node < CONFIG_T::n_out) {
            ap_uint<6> address = 0;
            for (unsigned pin = 0; pin < 6; pin++) {
                #pragma HLS UNROLL
                address[pin] = lut6_read(input, CONFIG_T::connections[node][pin]);
            }
            const ap_uint<1> bit = CONFIG_T::truth_tables[node][address];
            if (offset < 64) {
                first_word[offset] = bit;
            } else {
                second_word[offset - 64] = bit;
            }
        }
    }
    output[0] = first_word;
    if (TILE * 2 + 1 < CONFIG_T::output_banks) {
        output[1] = second_word;
    }
}

template<unsigned TILE, typename CONFIG_T,
         bool DONE = (TILE == CONFIG_T::n_tiles)>
struct lut6_dense_tiles;

template<unsigned TILE, typename CONFIG_T>
struct lut6_dense_tiles<TILE, CONFIG_T, false> {
    static void run(const ap_uint<64> *input, ap_uint<64> *output) {
        #pragma HLS INLINE
        lut6_dense_tile<TILE, CONFIG_T>(input, output + TILE * 2);
        lut6_dense_tiles<TILE + 1, CONFIG_T>::run(input, output);
    }
};

template<unsigned TILE, typename CONFIG_T>
struct lut6_dense_tiles<TILE, CONFIG_T, true> {
    static void run(const ap_uint<64> *, ap_uint<64> *) {}
};

template<class data_T, class result_T, typename CONFIG_T>
void lut6_dense_packed(ap_uint<64> input[(CONFIG_T::n_in + 63) / 64],
                       ap_uint<64> output[CONFIG_T::output_banks]) {
    lut6_dense_tiles<0, CONFIG_T>::run(input, output);
}

template<unsigned COUNT, class sum_T>
sum_T boolean_sum_128(const ap_uint<64> *input, unsigned start_index) {
    #pragma HLS INLINE
    static_assert(COUNT > 0 && COUNT <= 128, "Boolean sum tile must contain 1..128 bits");
    sum_T level0[128];
    sum_T level1[64];
    sum_T level2[32];
    sum_T level3[16];
    sum_T level4[8];
    sum_T level5[4];
    sum_T level6[2];
    #pragma HLS ARRAY_PARTITION variable=level0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=level1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=level2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=level3 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=level4 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=level5 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=level6 complete dim=0
    for (unsigned index = 0; index < 128; index++) {
        #pragma HLS UNROLL
        level0[index] = index < COUNT
            ? static_cast<sum_T>(lut6_read(input, start_index + index)) : sum_T(0);
    }
    for (unsigned index = 0; index < 64; index++) {
        #pragma HLS UNROLL
        level1[index] = level0[2 * index] + level0[2 * index + 1];
    }
    for (unsigned index = 0; index < 32; index++) {
        #pragma HLS UNROLL
        level2[index] = level1[2 * index] + level1[2 * index + 1];
    }
    for (unsigned index = 0; index < 16; index++) {
        #pragma HLS UNROLL
        level3[index] = level2[2 * index] + level2[2 * index + 1];
    }
    for (unsigned index = 0; index < 8; index++) {
        #pragma HLS UNROLL
        level4[index] = level3[2 * index] + level3[2 * index + 1];
    }
    for (unsigned index = 0; index < 4; index++) {
        #pragma HLS UNROLL
        level5[index] = level4[2 * index] + level4[2 * index + 1];
    }
    for (unsigned index = 0; index < 2; index++) {
        #pragma HLS UNROLL
        level6[index] = level5[2 * index] + level5[2 * index + 1];
    }
    return level6[0] + level6[1];
}

template<unsigned FLAT_TILE, class result_T, typename CONFIG_T>
void boolean_group_sum_tile(const ap_uint<64> *input, result_T &partial) {
    #pragma HLS INLINE off
    static const unsigned class_index = FLAT_TILE / CONFIG_T::tiles_per_class;
    static const unsigned class_tile = FLAT_TILE % CONFIG_T::tiles_per_class;
    static const unsigned class_offset = class_tile * CONFIG_T::tile_size;
    static const unsigned remaining = CONFIG_T::group_size - class_offset;
    static const unsigned count =
        remaining < CONFIG_T::tile_size ? remaining : CONFIG_T::tile_size;
    static const unsigned input_offset = class_index * CONFIG_T::group_size + class_offset;
    partial = boolean_sum_128<count, result_T>(input, input_offset);
}

template<unsigned FLAT_TILE, class result_T, typename CONFIG_T,
         bool DONE = (FLAT_TILE == CONFIG_T::classes * CONFIG_T::tiles_per_class)>
struct boolean_group_sum_tiles;

template<unsigned FLAT_TILE, class result_T, typename CONFIG_T>
struct boolean_group_sum_tiles<FLAT_TILE, result_T, CONFIG_T, false> {
    static void run(const ap_uint<64> *input, result_T *partial) {
        #pragma HLS INLINE
        boolean_group_sum_tile<FLAT_TILE, result_T, CONFIG_T>(input, partial[FLAT_TILE]);
        boolean_group_sum_tiles<FLAT_TILE + 1, result_T, CONFIG_T>::run(input, partial);
    }
};

template<unsigned FLAT_TILE, class result_T, typename CONFIG_T>
struct boolean_group_sum_tiles<FLAT_TILE, result_T, CONFIG_T, true> {
    static void run(const ap_uint<64> *, result_T *) {}
};

template<class data_T, class result_T, typename CONFIG_T>
void boolean_group_sum_packed(ap_uint<64> input[(CONFIG_T::n_in + 63) / 64],
                              result_T output[CONFIG_T::classes]) {
    static_assert(CONFIG_T::bank_width == 64 && CONFIG_T::tile_size == 128,
                  "Boolean GroupSum expects 128-bit local tiles");
    static_assert(CONFIG_T::tiles_per_class <= 8,
                  "Boolean GroupSum final tree supports up to 8 tiles");
    result_T partial[CONFIG_T::classes * CONFIG_T::tiles_per_class];
    #pragma HLS ARRAY_PARTITION variable=partial complete dim=0
    boolean_group_sum_tiles<0, result_T, CONFIG_T>::run(input, partial);
    for (unsigned class_index = 0; class_index < CONFIG_T::classes; class_index++) {
        #pragma HLS UNROLL
        result_T final0[8];
        result_T final1[4];
        result_T final2[2];
        #pragma HLS ARRAY_PARTITION variable=final0 complete dim=0
        #pragma HLS ARRAY_PARTITION variable=final1 complete dim=0
        #pragma HLS ARRAY_PARTITION variable=final2 complete dim=0
        for (unsigned tile = 0; tile < 8; tile++) {
            #pragma HLS UNROLL
            final0[tile] = tile < CONFIG_T::tiles_per_class
                ? partial[class_index * CONFIG_T::tiles_per_class + tile] : result_T(0);
        }
        for (unsigned index = 0; index < 4; index++) {
            #pragma HLS UNROLL
            final1[index] = final0[2 * index] + final0[2 * index + 1];
        }
        for (unsigned index = 0; index < 2; index++) {
            #pragma HLS UNROLL
            final2[index] = final1[2 * index] + final1[2 * index + 1];
        }
        output[class_index] = final2[0] + final2[1];
    }
}

} // namespace nnet

#endif
