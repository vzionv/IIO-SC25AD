`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: TalTech
// Engineer: Nikita Budovey
// 
// Create Date: 02/27/2023 06:23:17 PM 
// Module Name: PE
// Project Name: Systolic Array
// Target Devices: Artix 7 cpg236
// Tool Versions: Vivado 2022.1.2
// Description: Computational unit that sums and multiplies input values.
//
//////////////////////////////////////////////////////////////////////////////////


(*use_dsp = "yes"*)module PE #(
    parameter DATA_WIDTH  = 16,
    parameter MATRIX_SIZE = 2
) (
    input             [           7:0] number_iterations,
    input                              rstn,
    input                              clk,
    input                              en,
    input  reg signed [DATA_WIDTH-1:0] weights,
    input  reg signed [DATA_WIDTH-1:0] inputs,
    input  reg signed [DATA_WIDTH-1:0] eta,
    input  reg signed [           7:0] delta,
    input  reg signed [           7:0] beta,
    output signed     [DATA_WIDTH-1:0] next_pe_weights,
    output signed     [DATA_WIDTH-1:0] next_pe_inputs,
    output reg signed [DATA_WIDTH-1:0] accumulator_result
);
    localparam signed lev0 = 0;
    localparam signed lev1 = 6 * (2 ** DATA_WIDTH - 1) * (2 ** (DATA_WIDTH - 1) - 1);

    logic signed [DATA_WIDTH-1:0] inner_next_pe_inputs, inner_next_pe_weights;
    logic signed [2*DATA_WIDTH+2:0] temp_accumulator_result;
    logic signed [2*DATA_WIDTH-1:0] mult_result;
    logic signed [2*DATA_WIDTH+2:0] sum_in_wh;
    logic                           start_accumulating;

    assign mult_result = weights * inputs;
    // Multiplier #(
    //     .DATA_WIDTH(DATA_WIDTH)
    // ) multiplyer (
    //     .A(weights),
    //     .B(inputs),
    //     .O(mult_result)
    // );


    Accumulator #(
        .DATA_WIDTH(2 * DATA_WIDTH)
    ) contaioner (
        .number_iterations (number_iterations),
        .rstn              (rstn),
        .clk               (clk),
        .start_accumulating(start_accumulating),
        .input_data        (mult_result),
        .result            (sum_in_wh)
    );

    always_ff @(posedge clk) begin
        if (!rstn) begin
            inner_next_pe_inputs <= 0;
            inner_next_pe_weights <= 0;
            start_accumulating <= 0;
        end else if (en) begin
            start_accumulating <= 1;
            inner_next_pe_inputs <= inputs;
            inner_next_pe_weights <= weights;
        end
    end

    assign next_pe_weights = inner_next_pe_weights;
    assign next_pe_inputs  = inner_next_pe_inputs;

    always_comb begin
        if (delta > lev0) begin
            temp_accumulator_result = $signed((sum_in_wh + eta) << ((delta + 1)));
        end else if (delta < lev0) begin
            temp_accumulator_result = $signed((sum_in_wh + eta) >> ((1 - delta)));
        end else begin
            temp_accumulator_result = $signed((sum_in_wh + eta));
        end
    end

    //relu
    always_comb begin
        if (temp_accumulator_result < lev0) begin
            accumulator_result = 0;
        end else if (temp_accumulator_result > lev1) begin
            accumulator_result = lev1 >> (DATA_WIDTH - 1 + beta);
        end else begin
            accumulator_result = temp_accumulator_result >> (DATA_WIDTH - 1 + beta);
        end
    end


endmodule
