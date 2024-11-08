#include <iostream>

void initialize_matrix(float *data, int row, int col) {
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            data[r * col + c] = float(r + c);  // Accessing as a 1D array
        }
    }
}

void display_matrix(float *data, int row, int col) {
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            std::cout << data[r * col + c] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    // Matrix dimensions
    int a_row = 3, a_col = 4, b_row = 4, b_col = 5;

    // Create matrices using arrays
    float a[a_row][a_col], b[b_row][b_col];

    // Initialize values (pass as a pointer to the first element)
    initialize_matrix(&a[0][0], a_row, a_col);
    initialize_matrix(&b[0][0], b_row, b_col);

    // Displaying matrices for verification (optional)
    std::cout << "Matrix A:\n";
    display_matrix(&a[0][0], a_row, a_col);

    std::cout << "Matrix B:\n";
    display_matrix(&b[0][0], b_row, b_col);

    return 0;
}
