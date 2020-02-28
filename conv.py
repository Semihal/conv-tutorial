import numpy as np


class Conv:

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        """
        Applies a 2D convolution over an input.
        In the simplest case, the output value of the layer with input
            size (N, in_channels, H, W) and output (N, out_channels, H, W).
        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution. Default: 1
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # initialize parameters
        conv_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.parameters = np.random.random(conv_shape)
        # cache for backward
        self._cache = {}
        self._gradient = np.zeros(self.parameters.shape)

    def _make_empty_output(self, input_shape: tuple) -> np.ndarray:
        """
        Make empty output array.
        Args:
            input_shape (tuple): Input shape size.

        Returns (np.ndarray):
            Empty np.ndarray with new shape.
        """
        if len(input_shape) != 3:
            raise ValueError('The input value must be a 3-D array.')
        # get output shape
        channels = self.out_channels
        height = self._size_size(input_shape[1])
        width = self._size_size(input_shape[2])
        output_shape = (channels, height, width)
        # make empty output array
        dummy = np.empty(output_shape)
        return dummy

    def _size_size(self, side) -> int:
        """
        Calculate size size.
        Args:
            side (int): Current side size.

        Returns:
            New side size by formula:
                ((current_size - kernel_size) / stride) + 1
        """
        side_size = ((side - self.kernel_size) / self.stride) + 1
        return int(side_size)

    def forward(self, x):
        """
        Convolution pass.
        Args:
            x: Input, shape as like (input_channel, height, width)

        See Also:
            http://cs231n.github.io/convolutional-networks/#conv
        """
        self._cache['input'] = x
        output = self._make_empty_output(x.shape)
        channels, height, width = output.shape

        for index_row in range(height):
            for index_col in range(width):
                # input part shape as like kernel_size
                end_row = index_row + self.kernel_size
                start_col = index_col * self.stride
                end_col = start_col + self.kernel_size
                input_part = x[:, index_row:end_row, start_col:end_col]
                # flatten input and params
                flatten_input = input_part.flatten()
                flatten_params = self.parameters.reshape(self.out_channels, -1)
                cell_value = flatten_input.dot(flatten_params.T)
                # write value to cell
                output[:, index_row, index_col] = cell_value

        return output

    def backward(self, error):
        x = self._cache['input']

        # pass through the dimension of parameters
        for index_row in range(self.parameters.shape[-2]):
            for index_col in range(self.parameters.shape[-1]):
                # error.shape = output.shape
                end_row = index_row + error.shape[1]
                end_col = index_col + error.shape[2]
                # result is equal error multiplied by corresponding input
                conv_input = x[:, index_row:end_row, index_col:end_col]
                # flatten to [input_channels, height x width]
                flatten_input = conv_input.reshape(x.shape[0], -1)
                # flatten to [output_channels, height_out x width_out]
                flatten_error = error.reshape(self.out_channels, -1)
                value = flatten_error.dot(flatten_input.T)
                # write value for corresponding parameter
                self._gradient[:, :, index_row, index_col] = value


if __name__ == '__main__':
    np.random.seed(0)

    # init consts
    INPUT_SHAPE = [2, 3, 3]
    INPUT_CHANNELS = INPUT_SHAPE[0]
    OUT_CHANNELS = 3
    KERNEL_SIZE = 2
    STRIDES = 1

    # generate dummy input and ones target
    dummy_input = (np.random.random(INPUT_SHAPE) * 10).astype(np.int)
    # random input
    conv = Conv(INPUT_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDES)
    result = conv.forward(dummy_input)
    conv.backward(result)
