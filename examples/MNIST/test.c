#include <raylib.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "neuralnetwork.h"
#include "mat.h"


uint8_t get_label_from_mat(mat *mat)
{
  size_t i;
  double prob;
  size_t max_label;
  double max_prob = -INFINITY;
  for (i=0; i < 10; i++)
  {
    prob = mat->data[i];
    if (prob > max_prob)
    {
      max_prob = prob;
      max_label = i;
    }
  }
  return max_label;
}

void paint_board(Color *board, int x, int y, int pixels)
{
  if (x < 0 || x >= pixels || y < 0 || y >= pixels)
    return;
  board[y*pixels + x] = WHITE;
}


int main(void) 
{
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(100, 100, "MNIST Dataset");
  SetTargetFPS(60);

  neural_network nn;
  nn_load(&nn, "model");
  
  int width;
  int height;

  const int pixels = 28;

  const float board_width = 0.5f;
  const float board_x = 0.0f;
  const float board_y = 0.0f;
  int start_x;
  int start_y;
  int cell_size;

  uint8_t prediction = 0;
  
  Color board[pixels * pixels];
  memset(board, 0x00, sizeof(int) * pixels * pixels);

  int x, y;
  int mouse_x, mouse_y;
  while (!WindowShouldClose())
  {
    width = GetScreenWidth();
    height = GetScreenHeight();
    cell_size = (int)(width * board_width) / pixels;
    start_x = width * board_x;
    start_y = height * board_y;
    
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
    {
      mouse_x = GetMouseX();
      mouse_y = GetMouseY();
      x = (int)((mouse_x - start_x) / cell_size);
      y = (int)((mouse_y - start_y) / cell_size);
      paint_board(board, x, y, pixels);
      paint_board(board, x+1, y, pixels);
      paint_board(board, x, y+1, pixels);
      paint_board(board, x+1, y+1, pixels);
    }

    if (IsKeyPressed(KEY_C))
      memset(board, 0x00, sizeof(int) * pixels * pixels);

    if (IsKeyPressed(KEY_P))
    {
      double d_board[pixels * pixels];
      for (x=0; x < pixels * pixels; x++)
        d_board[x] = board[x].r / 255.0;
      mat input;
      mat_create_from_buffer(&input, pixels * pixels, 1, d_board);
      nn_feedforward(&nn, &input);
      prediction = get_label_from_mat(&nn.layers[nn.size-1]);
    }
    
    BeginDrawing();
    ClearBackground(BLACK);

    DrawText(TextFormat("Prediction: %d", prediction), cell_size * pixels, 0, 20, WHITE);
    
    DrawRectangleLines(
      start_x, 
      start_y, 
      pixels * cell_size, 
      pixels * cell_size, 
      GRAY
    );

    for (y=0; y < pixels; y++)
    for (x=0; x < pixels; x++)
    {
      DrawRectangle(
        start_x + x*cell_size, 
        start_y + y*cell_size, 
        cell_size, cell_size, 
        board[y*pixels + x]
      );
    }

    EndDrawing();
  }

  CloseWindow();

  return 0;
}

