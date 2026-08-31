// ---------------------------------------------------------
// Phase 40: FPGA Hardware Synthesis (Verilog RTL)
// ---------------------------------------------------------
// This module describes the digital logic gates required to execute a 
// Breadth-First Search (BFS) directly in silicon, completely bypassing the CPU.

module bfs_accelerator (
    input wire clk,
    input wire rst,
    input wire start,
    
    // Memory Interface (AXI/BRAM)
    input wire [31:0] vertex_data,
    output reg [31:0] memory_addr,
    output reg memory_read_en,
    
    // Status
    output reg done
);

    // BFS State Machine
    localparam IDLE         = 3'b000;
    localparam FETCH_NODE   = 3'b001;
    localparam FETCH_EDGES  = 3'b010;
    localparam UPDATE_DIST  = 3'b011;
    localparam DONE         = 3'b100;
    
    reg [2:0] state, next_state;
    reg [31:0] current_vertex;
    reg [31:0] edge_count;

    // FIFO Queue for Frontier (Synthesizes to BRAM on FPGA)
    reg [31:0] frontier_queue [0:1023];
    reg [9:0] head;
    reg [9:0] tail;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            head <= 0;
            tail <= 0;
            done <= 0;
        end else begin
            state <= next_state;
        end
    end

    // Combinational State Transition Logic
    always @(*) begin
        next_state = state;
        memory_read_en = 0;
        
        case (state)
            IDLE: begin
                if (start) begin
                    next_state = FETCH_NODE;
                    // Seed the frontier (e.g., vertex 0)
                    frontier_queue[tail] = 0;
                    tail = tail + 1;
                end
            end
            
            FETCH_NODE: begin
                if (head == tail) begin
                    next_state = DONE;
                end else begin
                    current_vertex = frontier_queue[head];
                    head = head + 1;
                    
                    // Request CSR row_offsets from memory
                    memory_addr = current_vertex * 4; 
                    memory_read_en = 1;
                    next_state = FETCH_EDGES;
                end
            end
            
            FETCH_EDGES: begin
                // In a real design, wait for memory valid signal
                // Read edges and enqueue them if unvisited
                next_state = UPDATE_DIST;
            end
            
            UPDATE_DIST: begin
                // Update distance array in BRAM
                next_state = FETCH_NODE;
            end
            
            DONE: begin
                done = 1;
            end
        endcase
    end

endmodule
