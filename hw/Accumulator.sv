`timescale 1ns / 1ps

module Accumulator #(
    parameter DATA_WIDTH = 4
) (
    input         [           7:0] number_iterations,
    input                          rstn,
    input                          clk,
    input                          start_accumulating,
    input  signed [DATA_WIDTH-1:0] input_data,
    output signed [DATA_WIDTH+2:0] result
);

    logic signed [DATA_WIDTH-1:0] mult_result;
    logic signed [DATA_WIDTH+2:0] sum;
    logic signed [DATA_WIDTH+2:0] sum_results;
    logic        [           7:0] counter;
    // logic                           cout;

    assign sum = sum_results + mult_result;
    //  Adder #(.DATA_WIDTH(DATA_WIDTH)) add(
    // .a              (sum_results), 
    // .b              (mult_result),
    // .cin            (0),
    // .sum            (sum),
    // .cout           (cout));


    always_ff @(posedge clk) begin

        if (!rstn) begin
            mult_result <= 0;
            sum_results <= 0;
            counter <= 0;
        end else if (start_accumulating) begin

            if (counter == number_iterations + 1) begin

                mult_result <= 0;
                sum_results <= sum;
            end else begin

                mult_result <= input_data;
                sum_results <= sum;
                counter <= counter + 1;

            end

        end

    end

    assign result = sum_results;

endmodule
