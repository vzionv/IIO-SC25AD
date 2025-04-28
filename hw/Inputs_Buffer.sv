`timescale 1ns / 1ps

module Inputs_Buffer #(
    parameter MATRIX_SIZE = 2,
    parameter DATA_WIDTH  = 16
) (
    input  signed [DATA_WIDTH-1:0] input_data,
    input                          rstn,
    input                          clk,
    input                          writing_signal,
    input                          weights_data_ready,
    output                         data_ready,
    output signed [DATA_WIDTH-1:0] row_array         [MATRIX_SIZE-1:0]
);
    logic signed [DATA_WIDTH-1:0] inputs_matrix[(MATRIX_SIZE + MATRIX_SIZE)-2:0][MATRIX_SIZE - 1:0];
    logic signed [DATA_WIDTH-1:0] prepared_output_data[MATRIX_SIZE-1:0];
    logic [7:0] row, colum;
    logic [7:0] out_row_index;
    logic       inner_data_ready;


    always_ff @(posedge clk) begin
        if (!rstn) begin
            row <= 0;
            colum <= 0;
            inner_data_ready <= 0;
            out_row_index <= 0;

            for (int i = 0; i < MATRIX_SIZE; i++) begin
                prepared_output_data[i] <= 0;
            end

            for (int i = 0; i < (MATRIX_SIZE + MATRIX_SIZE) - 1; i++) begin
                for (int j = 0; j < MATRIX_SIZE; j++) begin
                    inputs_matrix[i][j] <= 0;
                end
            end
        end else if (row == MATRIX_SIZE) begin
            inner_data_ready <= 1;
            if (weights_data_ready && out_row_index < (MATRIX_SIZE + MATRIX_SIZE) - 1) begin
                for (int i = 0; i < MATRIX_SIZE; i++) begin
                    prepared_output_data[i] <= inputs_matrix[out_row_index][i];
                end
                out_row_index <= out_row_index + 1;
            end else if (weights_data_ready) begin
                for (int i = 0; i < MATRIX_SIZE; i++) begin
                    prepared_output_data[i] <= 0;
                end
                out_row_index <= 0;
                inner_data_ready <= 0;
                row   <= 0;
                colum <= 0;
            end
        end else if (writing_signal) begin
            if (colum == MATRIX_SIZE - 1) begin
                row   <= row + 1;
                colum <= 0;
            end else begin
                colum <= colum + 1;
            end
            inputs_matrix[colum+row][colum] <= input_data;
        end
    end

    assign row_array  = prepared_output_data;
    assign data_ready = inner_data_ready;

endmodule
