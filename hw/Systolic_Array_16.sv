`timescale 1ns / 1ps

module Systolic_Array_16 #(
    parameter MATRIX_SIZE = 5,
    parameter DATA_WIDTH  = 8
) (
    input                              rstn,
    input                              clk,
    input  signed     [DATA_WIDTH-1:0] inputs,
    input  signed     [DATA_WIDTH-1:0] weights,
    input                              start,
    output reg signed [DATA_WIDTH-1:0] result_row[MATRIX_SIZE - 1:0],
    output reg                         finished
);
    logic signed [DATA_WIDTH-1:0] inputs_array    [3:0];
    logic signed [DATA_WIDTH-1:0] weights_array   [3:0];
    logic signed [DATA_WIDTH-1:0] result_row_array[3:0] [MATRIX_SIZE - 1:0];
    logic                         finished_array  [3:0];
    logic                         writing_signal  [3:0];
    genvar i, j;

    generate
        for (i = 0; i < 4; i++) begin
            Systolic_Array #(
                .MATRIX_SIZE(MATRIX_SIZE),
                .DATA_WIDTH (DATA_WIDTH)
            ) sa (
                .rstn          (rstn),
                .clk           (clk),
                .writing_signal(writing_signal[i]),
                .inputs        (inputs_array[i]),
                .weights       (weights_array[i]),
                .result_row    (result_row_array[i]),
                .finished      (finished_array[i])
            );
        end
    endgenerate

    always_comb begin
        finished = 1'b1;
        for (int i = 0; i < 4; i++) begin
            finished = finished & finished_array[i];
            for (int j = 0; j < MATRIX_SIZE; j++) begin
                result_row[j] = result_row[j] + result_row_array[i][j];
            end
        end
    end

    reg [7:0] counter;
    always_ff @(posedge clk) begin
        if (!rstn) begin
            for (int i = 0; i < 4; i++) begin
                writing_signal[i] <= 0;
                inputs_array[i]   <= 0;
                weights_array[i]  <= 0;
            end
            counter <= 0;
        end else if (start) begin
            if (counter == 15) begin
                counter <= 0;
            end else begin
                counter <= counter + 1;
                writing_signal[counter] <= 1'b1;
                inputs_array[counter] <= inputs;
                weights_array[counter] <= weights;
            end
        end else begin
            for (int i = 0; i < 4; i++) begin
                writing_signal[i] <= 0;
                inputs_array[i]   <= 0;
                weights_array[i]  <= 0;
            end
            counter <= 0;
        end
    end
endmodule

