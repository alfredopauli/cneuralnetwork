#include <raylib.h>
#include <string.h>


int main(void) 
{
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(100, 100, "MNIST Dataset");
  SetTargetFPS(60);
  
  int width;
  int height;

  const int pixels = 28;
  const float board_width = 0.5f;
  float cell_size;
  
  char board[pixels * pixels];
  memset(board, '\0', sizeof(int) * pixels * pixels);

  int x, y;
  while (!WindowShouldClose())
  {
    BeginDrawing();
    ClearBackground(BLACK);
    
    width = GetScreenWidth();
    height = GetScreenHeight();
    cell_size = width * board_width / pixels;
    
    for (x=0; x < pixels; x++)
    for (y=0; y < pixels; y++)
    {
      DrawRectangle(x*cell_size, y*cell_size, cell_size, cell_size, board[y*pixels + x]);
    }

    EndDrawing();
  }

  CloseWindow();

  return 0;
}

