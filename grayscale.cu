
__global__
void colortoGrayscaleConvertion(unsigned char *Pout,
                                unsigned char *Pin, int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if(col < width && row < height) {
        int grayOffset = row * width + col;

        int rgbOffset = grayOffset*3;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[grayOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}