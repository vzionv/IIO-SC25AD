`timescale 1ns / 1ps


module Systolic_Array #(
    parameter MATRIX_SIZE = 5,
    parameter DATA_WIDTH  = 8
) (
    input                          rstn,
    input                          clk,
    input                          writing_signal,
    input  signed [DATA_WIDTH-1:0] inputs,
    input  signed [DATA_WIDTH-1:0] weights,
    output reg signed [DATA_WIDTH-1:0] result_row    [MATRIX_SIZE - 1:0],
    output reg                     finished
);

    // logic signed    [DATA_WIDTH-1:0]    inner_result;
    logic        [           7:0] counter;
    logic signed [DATA_WIDTH-1:0] next_inputs        [    MATRIX_SIZE:0] [MATRIX_SIZE - 1:0];
    logic signed [DATA_WIDTH-1:0] next_weights       [MATRIX_SIZE - 1:0] [    MATRIX_SIZE:0];
    logic signed [DATA_WIDTH-1:0] array_results      [MATRIX_SIZE - 1:0] [MATRIX_SIZE - 1:0];
    logic                         weights_data_ready;
    logic                         inputs_data_ready;
    logic signed [DATA_WIDTH-1:0] colum              [MATRIX_SIZE - 1:0];
    logic signed [DATA_WIDTH-1:0] row                [MATRIX_SIZE - 1:0];
    // logic [DATA_WIDTH-1:0]    result [MATRIX_SIZE - 1:0][MATRIX_SIZE - 1:0];
    logic                         o_wr_en;
    reg          [           7:0] output_row_index;
    reg                           pe_en;
    genvar i, j;

    generate
        for (i = 0; i < MATRIX_SIZE; i++) begin
            for (j = 0; j < MATRIX_SIZE; j++) begin
                localparam number_iterations = MATRIX_SIZE + j + (MATRIX_SIZE * i);
                PE #(
                    .DATA_WIDTH(DATA_WIDTH)
                ) pe (
                    .number_iterations (number_iterations[7:0]),
                    .rstn              (rstn),
                    .clk               (clk),
                    .en                (pe_en),
                    .weights           (next_weights[i][j]),
                    .inputs            (next_inputs[i][j]),
                    .next_pe_weights   (next_weights[i][j+1]),
                    .next_pe_inputs    (next_inputs[i+1][j]),
                    .accumulator_result(array_results[i][j]),
                    .eta               (8'd0),
                    .delta             (8'd0),
                    .beta              (8'd0)
                );
            end

        end
    endgenerate

    Weights_Buffer #(
        .MATRIX_SIZE(MATRIX_SIZE),
        .DATA_WIDTH ((DATA_WIDTH))
    ) weights_register (
        .rstn             (rstn),
        .clk              (clk),
        .writing_signal   (writing_signal),
        .inputs_data_ready(inputs_data_ready),
        .data_ready       (weights_data_ready),
        .input_data       (weights),
        .colum_array      (colum)
    );


    Inputs_Buffer #(
        .MATRIX_SIZE(MATRIX_SIZE),
        .DATA_WIDTH ((DATA_WIDTH))
    ) inputs_register (
        .rstn              (rstn),
        .clk               (clk),
        .writing_signal    (writing_signal),
        .weights_data_ready(weights_data_ready),
        .data_ready        (inputs_data_ready),
        .input_data        (inputs),
        .row_array         (row)
    );

    Outputs_Buffer #(
        .MATRIX_SIZE(MATRIX_SIZE),
        .DATA_WIDTH ((DATA_WIDTH))
    ) outputs_register (
        .rstn      (rstn),
        .clk       (clk),
        .wr_en     (o_wr_en),
        .input_data(result_row)
    );


    always_ff @(posedge clk) begin
        if (!rstn) begin
            finished <= 0;
            pe_en <= 0;
            counter <= 0;
            output_row_index <= 0;
            o_wr_en <= 0;
            for (int i = 0; i < MATRIX_SIZE; i++) begin
                next_inputs[0][i]  <= 0;
                next_weights[i][0] <= 0;
            end
            for (int i = 0; i < MATRIX_SIZE; i++) begin
                result_row[i] <= 0;
            end
        end else if (counter >= MATRIX_SIZE + MATRIX_SIZE) begin
            if (output_row_index == MATRIX_SIZE) begin
                finished <= 1'b1;
                output_row_index <= 0;
                counter <= 0;
                o_wr_en <= 0;
                for (int i = 0; i < MATRIX_SIZE; i++) begin
                    result_row[i] <= 0;
                end
            end else if (counter == MATRIX_SIZE + MATRIX_SIZE + 1) begin
                o_wr_en <= 1;
                counter <= counter + 1;
            end else if (counter >= MATRIX_SIZE + MATRIX_SIZE + 2) begin
                for (int i = 0; i < MATRIX_SIZE; i++) begin
                    result_row[i] <= array_results[output_row_index][i];
                end
                output_row_index <= output_row_index + 1;
                counter <= counter + 1;
            end else begin
                counter <= counter + 1;
            end
            for (int i = 0; i < MATRIX_SIZE; i++) begin
                next_inputs[0][i]  <= 0;
                next_weights[i][0] <= 0;
            end

        end
        else if (weights_data_ready && inputs_data_ready) begin
            pe_en <= 1;
            next_inputs[0][MATRIX_SIZE-1:0] <= row[MATRIX_SIZE-1:0];

            for (int i = 0; i < MATRIX_SIZE; i++) begin
                next_weights[i][0] <= colum[i];
            end

            counter <= counter + 1;

        end else begin
            pe_en <= 0;
            finished <= 0;
            counter <= 0;
            output_row_index <= 0;
            o_wr_en <= 0;
            for (int i = 0; i < MATRIX_SIZE; i++) begin
                next_inputs[0][i]  <= 0;
                next_weights[i][0] <= 0;
            end
            for (int i = 0; i < MATRIX_SIZE; i++) begin
                result_row[i] <= 0;
            end
        end
    end

endmodule
