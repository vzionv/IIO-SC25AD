`timescale 1ns / 1ps


module Outputs_Buffer #(
    parameter MATRIX_SIZE = 2,
    parameter DATA_WIDTH  = 16
) (
    input  signed [DATA_WIDTH-1:0] input_data         [MATRIX_SIZE-1:0],
    input                          rstn,
    input                          clk,
    input                          wr_en
);
    logic signed [DATA_WIDTH-1:0] outputs_matrix[MATRIX_SIZE - 1:0][MATRIX_SIZE - 1:0];
    logic [7:0] row;
    always_ff @(posedge clk) begin
        if (!rstn) begin
            row <= 0;
            for (int i = 0; i < MATRIX_SIZE; i++) begin
                for (int j = 0; j < MATRIX_SIZE; j++) begin
                    outputs_matrix[i][j] <= 0;
                end
            end
        end else if (wr_en) begin
            if (row == MATRIX_SIZE - 1) begin
                row   <= 0;
            end else begin
                row <= row + 1;
            end
            outputs_matrix[row] <= input_data;

        end
    end

endmodule
