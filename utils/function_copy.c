
arm_cmsis_nn_status arm_fully_connected_s8_test(const cmsis_nn_context *ctx,
                                           const cmsis_nn_fc_params *fc_params,
                                           const cmsis_nn_per_tensor_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int8_t *input,
                                           const cmsis_nn_dims *filter_dims,
                                           const int8_t *kernel,
                                           const cmsis_nn_dims *bias_dims,
                                           const int32_t *bias,
                                           const cmsis_nn_dims *output_dims,
                                           int8_t *output)
{
    (void)bias_dims;

    int32_t batch_cnt = input_dims->n;

#if defined(ARM_MATH_MVEI)
    if (ctx->buf == NULL)
    {
        return (ARM_CMSIS_NN_ARG_ERROR);
    }
#endif

    const int32_t *kernel_sum = (const int32_t *)ctx->buf;

    while (batch_cnt)
    {

        arm_nn_vec_mat_mult_t_s8_test(input,
                                 kernel,
                                 kernel_sum,
                                 bias,
                                 output,
                                 fc_params->input_offset,
                                 fc_params->output_offset,
                                 quant_params->multiplier,
                                 quant_params->shift,
                                 filter_dims->n, /* col_dim or accum_depth */
                                 output_dims->c, /* row_dim or output_depth */
                                 fc_params->activation.min,
                                 fc_params->activation.max,
                                 1L,
                                 fc_params->filter_offset);

        input += filter_dims->n;
        output += output_dims->c;
        batch_cnt--;
    }
    return (ARM_CMSIS_NN_SUCCESS);
}

arm_cmsis_nn_status arm_nn_vec_mat_mult_t_s8_test(const int8_t *lhs,
                                             const int8_t *rhs,
                                             const int32_t *kernel_sum,
                                             const int32_t *bias,
                                             int8_t *dst,
                                             const int32_t lhs_offset,
                                             const int32_t dst_offset,
                                             const int32_t dst_multiplier,
                                             const int32_t dst_shift,
                                             const int32_t rhs_cols,
                                             const int32_t rhs_rows,
                                             const int32_t activation_min,
                                             const int32_t activation_max,
                                             const int32_t address_offset,
                                             const int32_t rhs_offset)
{
        (void)kernel_sum;

        const int32_t row_loop_cnt = rhs_rows / 3;

        for (int32_t i_row_loop_cnt = 0; i_row_loop_cnt < row_loop_cnt; i_row_loop_cnt++)
        {
            const int8_t *lhs_ptr = lhs;
            const int8_t *rhs_ptr_0 = &rhs[0];
            const int8_t *rhs_ptr_1 = &rhs[rhs_cols];
            const int8_t *rhs_ptr_2 = &rhs[rhs_cols * 2];

            int32_t res00 = 0;
            int32_t res01 = 0;
            int32_t res02 = 0;
            if (bias)
            {
                res00 = *bias++;
                res01 = *bias++;
                res02 = *bias++;
            }
            for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
            {
                const int32_t rhs_value0 = (int8_t)*rhs_ptr_0;
                const int32_t rhs_value1 = (int8_t)*rhs_ptr_1;
                const int32_t rhs_value2 = (int8_t)*rhs_ptr_2;
                const int32_t lhs_value = (int8_t)*lhs_ptr + lhs_offset;

                res00 += lhs_value * rhs_value0;
                res01 += lhs_value * rhs_value1;
                res02 += lhs_value * rhs_value2;

                ++rhs_ptr_0;
                ++rhs_ptr_1;
                ++rhs_ptr_2;
                ++lhs_ptr;
            }
            
            // test
            if(i_row_loop_cnt == 0)
            {
                printf("%d\n", res00);
            }

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);
            res01 = arm_nn_requantize(res01, dst_multiplier, dst_shift);
            res02 = arm_nn_requantize(res02, dst_multiplier, dst_shift);
            
            // test
            if(i_row_loop_cnt == 0)
            {
                printf("%d\n", res00);
            }

            // Add offset
            res00 += dst_offset;
            res01 += dst_offset;
            res02 += dst_offset;

            // Clamp the result
            res00 = MAX(res00, activation_min);
            res00 = MIN(res00, activation_max);
            res01 = MAX(res01, activation_min);
            res01 = MIN(res01, activation_max);
            res02 = MAX(res02, activation_min);
            res02 = MIN(res02, activation_max);

            *dst = (int8_t)res00;
            *(dst + address_offset) = (int8_t)res01;
            *(dst + 2 * address_offset) = (int8_t)res02;
            dst += 3 * address_offset;

            rhs += 3 * rhs_cols;
        }

        const int loop_cnt = rhs_rows % 3;

        for (int32_t i_loop_cnt = 0; i_loop_cnt < loop_cnt; i_loop_cnt++)
        {
            const int8_t *lhs_ptr = &lhs[0];
            const int8_t *rhs_ptr = &rhs[0];

            int32_t res00 = 0;
            if (bias)
            {
                res00 = *bias++;
            }

            for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
            {
                int32_t rhs_value0 = (int8_t)rhs_ptr[0];
                int32_t lhs_value = (int8_t)lhs_ptr[0] + lhs_offset;

                res00 += lhs_value * rhs_value0;

                ++rhs_ptr;
                ++lhs_ptr;
            }

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);

            // Add offset
            res00 += dst_offset;

            // Clamp the result
            res00 = MAX(res00, activation_min);
            res00 = MIN(res00, activation_max);

            *dst = (int8_t)res00;
            dst += address_offset;
            rhs += rhs_cols;
        }
    }