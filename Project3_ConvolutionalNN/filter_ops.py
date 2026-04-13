'''filter_ops.py
Implements the convolution and max pooling operations.
Applied to images and other data represented as an ndarray.
Duilio Lucio, Vivian Hu
CS343: Neural Networks
Project 3: Convolutional neural networks
'''
import numpy as np


def conv2_gray(img, kers, verbose=True):
    '''Does a 2D convolution operation on GRAYSCALE `img` using kernels `kers`.
    Uses 'same' boundary conditions.

    Parameters:
    -----------
    img: ndarray. Grayscale input image to be filtered. shape=(height img_y (px), width img_x (px))
    kers: ndarray. Convolution kernels. shape=(Num kers, ker_sz (px), ker_sz (px))
        NOTE: Kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    filteredImg: ndarray. `img` filtered with all the `kers`. shape=(Num kers, img_y, img_x)

    Hints:
    -----------
    - Remember to flip your kernel since this is convolution!
    - Be careful of off-by-one errors, especially in setting up your loops. In particular, you
    want to align your convolution so that it starts aligned with the top-left corner of your
    padded image and iterate until the right/bottom sides of the kernel fall in the last pixel on
    the right/bottom sides of the padded image.
    - Use the 'same' padding formula for compute the necessary amount of padding to have the output
    image have the same spatial dimensions as the input.
    - I suggest using indexing/assignment to 'frame' your input image into the padded one.
    '''
    img_y, img_x = img.shape
    n_kers, ker_y, ker_x = kers.shape

    if verbose:
        print(f'img_x={img_y}, img_y={img_x}')
        print(f'n_kers={n_kers}, ker_y={ker_y}, ker_x={ker_x}')

    if ker_x != ker_y:
        print('Kernels must be square!')
        return
    
    # Same padding calculation, makes output dimensions match input dimensions
    pad = ker_y // 2 # Zero Padding Formula = [(kerSz - 1) / 2]
    
    # Padded image, create canvas of zeros and place image in the middle of the canvas like a frame
    padded_image = np.zeros((img_y + 2 * pad, img_x + 2 * pad)) # 2 * pad to add border on both sides, left+right & top+bottom
    padded_image[pad : pad + img_y , pad : pad + img_x] = img # places image right after top of border(pad) and stop once reaching height of image
    
    # Output array
    filteredImg = np.zeros((n_kers, img_y, img_x)) # apply multiple kernals, n_kers is number of images in stack & each img. has same dimensions(y, x) as output
    
    # Convolution loop
    for k in range(n_kers):
        # Flip kernel
        # indexing [::-1. ::-1] flips image both vertically and horizontally
        curr_ker = kers[k][::-1, ::-1] # kers[k] pulls k-th individual 2D kernal from 3D block(a tensor, kers) 
        
        # Iterate through rows(from 0 - img_y) and cols.(from 0 - img_x)
        for y in range(img_y):
            for x in range(img_x):
                # extract window by slicing padded image
                # window starts at (y, x) and has shape of kernal
                window = padded_image[y : y + ker_y, x : x + ker_y] # looks at small neighboorhoods of pixels, y & x is top-left corner where kernel is sitting, then cuts padded img.(same size as kernal) and looks at it 
                
                # dot product (multiply element wise and sum)
                filteredImg[k, y, x] = np.sum(window * curr_ker) # multiples pixel vals. in window by weights in kernal, if kernal is high pixel is emphasized if its zero then its ignored
        if verbose:
            print(f'Finished kernal {k+1}/{n_kers}')
            
    return filteredImg

def conv2(img, kers, verbose=True):
    '''Does a 2D convolution operation on COLOR or grayscale `img` using kernels `kers`.
    Uses 'same' boundary conditions.

    Parameters:
    -----------
    img: ndarray. Input image to be filtered. shape=(N_CHANS, height img_y (px), width img_x (px))
        where n_chans is 1 for grayscale images and 3 for RGB color images
    kers: ndarray. Convolution kernels. shape=(Num filters, ker_sz (px), ker_sz (px))
        NOTE: Each kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    filteredImg: ndarray. `img` filtered with all `kers`. shape=(Num filters, N_CHANS, img_y, img_x)

    What's new:
    -----------
    - N_CHANS, see above.

    Hints:
    -----------
    - You should not need more for loops than you have in `conv2_gray`.
    - When summing inside your nested loops, keep in mind the keepdims=True parameter of np.sum and
    be aware of which axes you are summing over. If you use keepdims=True, you may want to remove
    singleton dimensions.
    '''
    n_chans, img_y, img_x = img.shape
    n_kers, ker_x, ker_y = kers.shape

    if verbose:
        print(f'n_chan={n_chans}, img_x={img_y}, img_y={img_x}')
        print(f'n_kers={n_kers}, ker_x={ker_x}, ker_y={ker_y}')

    if ker_x != ker_y:
        print('Kernels must be square!')
        return
    
    # Padding and initializing output
    pad = ker_y // 2
    filteredImg = np.zeros((n_kers, n_chans, img_y, img_x))
    
    # Pad input image, only pad height & weight (axis 1, axis 2), channel axis(axis 0) remains unpadded
    padded_img = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), mode = 'constant')

    # Convolution
    for k in range(n_kers):
        # Flip kernel for convolution logic
        curr_ker = kers[k, ::-1, ::-1]
        for y in range(img_y):
            for x in range(img_x):
                # Slice window across all channels (:), window shape: (n_chans, ker_y, ker_x)
                window = padded_img[:, y : y + ker_y, x : x + ker_x]
                
                # elem. wise multiplcation (broadcasting), (n_chans, ker_y, ker_x) * (ker_y, ker_x) -> (n_chans, ker_y, ker_x)
                conv_sum = np.sum(window * curr_ker, axis=(1, 2), keepdims=True)
                
                # Remove "singleton" dimensions, will now fit into (n_chans, ) slot in output
                filteredImg[k, :, y, x] = conv_sum.squeeze()
        if verbose:
            print(f'Finished filtering with kernal {k+1}/{n_kers}')
    return filteredImg
                

def conv2nn(imgs, kers, bias, verbose=True):
    '''General 2D convolution operation suitable for a convolutional layer of a neural network.
    Uses 'same' boundary conditions.

    Parameters:
    -----------
    imgs: ndarray. Input IMAGES to be filtered. shape=(BATCH_SZ, n_chans, img_y, img_x)
        where batch_sz is the number of images in the mini-batch
        n_chans is 1 for grayscale images and 3 for RGB color images
    kers: ndarray. Convolution kernels. shape=(n_kers, N_CHANS, ker_sz, ker_sz)
        NOTE: Each kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
    bias: ndarray. Bias term used in the neural network layer. Shape=(n_kers,)
        i.e. there is a single bias per filter. Convolution by the c-th filter gets the c-th
        bias term added to it.
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    What's new (vs conv2):
    -----------
    - Multiple images (mini-batch support)
    - Kernels now have a color channel dimension too
    - Collapse (sum) over color channels when computing the returned output images
    - A bias term

    Returns:
    -----------
    output: ndarray. `imgs` filtered with all `kers`. shape=(BATCH_SZ, n_kers, img_y, img_x)

    Hints:
    -----------
    - You may need additional loop(s).
    - Summing inside your loop can be made simpler compared to conv2.
    - Adding the bias should be easy.
    '''
    batch_sz, n_chans, img_y, img_x = imgs.shape
    n_kers, n_ker_chans, ker_y, ker_x = kers.shape

    if verbose:
        print(f'batch_sz={batch_sz}, n_chan={n_chans}, img_x={img_y}, img_y={img_x}')
        print(f'n_kers={n_kers}, n_ker_chans={n_ker_chans}, ker_y={ker_y}, ker_x={ker_x}')

    if ker_x != ker_y:
        print('Kernels must be square!')
        return

    if n_chans != n_ker_chans:
        print('Number of kernel channels doesnt match input num channels!')
        return

    # output array
    output = np.zeros((batch_sz, n_kers, img_y, img_x))
    
    # Padding for 4D array : (Batch, Channel, Height, Width)
    pad = ker_y // 2
    padded_img = np.pad(imgs, ((0, 0), (0,0), (pad, pad), (pad, pad)), mode = 'constant')
    
    # Convolution
    for k in range(n_kers):
        curr_ker = kers[k, :, ::-1, ::-1] # flip kernal
        for y in range(img_y):
            for x in range(img_x):
                # slice window across all images in batch and ALL channels
                window = padded_img[:, :, y : y + ker_y, x : x + ker_x]
                # element wise mult & sum across, collapse dims for 1 val
                conv_sum = np.sum(window * curr_ker, axis=(1, 2, 3))
                # assign every image in batch its own map val.
                output[:, k, y, x] = conv_sum
    # Add bias, reshape bias across Batch, Y, and X
    output += bias[np.newaxis, :, np.newaxis, np.newaxis]
    return output 
    


def get_pooling_out_shape(img_dim, pool_size, strides):
    '''Computes the size of the output of a max pooling operation along one spatial dimension.

    Parameters:
    -----------
    img_dim: int. Either img_y or img_x
    pool_size: int. Size of pooling window in one dimension: either x or y (assumed the same).
    strides: int. Size of stride when the max pooling window moves from one position to another.

    Returns:
    -----------
    int. The size in pixels of the output of the image after max pooling is applied, in the dimension
        img_dim.
    '''
    pass


def max_pool(inputs, pool_size=2, strides=1, verbose=True):
    ''' Does max pooling on inputs. Works on single grayscale images, so somewhat comparable to
    `conv2_gray`.

    Parameters:
    -----------
    inputs: Input to be filtered. shape=(height img_y, width img_x)
    pool_size: int. Pooling window extent in both x and y.
    strides: int. How many "pixels" in x and y to skip over between successive max pooling operations
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    outputs: Input filtered with max pooling op. shape=(out_y, out_x)
        NOTE: out_y, out_x determined by the output shape formula. The input spatial dimensions are
        not preserved (unless pool_size=1...but what's the point of that? :)

    NOTE: There is no padding in the max-pooling operation.

    Hints:
    -----------
    - You should be able to heavily leverage the structure of your conv2_gray code here
    - Instead of defining a kernel, indexing strategically may be helpful
    - You may need to keep track of and update indices for both the input and output images
    - Overall, this should be a simpler implementation than `conv2_gray`
    '''
    img_y, img_x = inputs.shape
    # Calculating output dims. , formula : out = ((in - pool_size) // strides) + 1 
    out_y = (img_y - pool_size) // strides + 1
    out_x = (img_x - pool_size) // strides + 1
    
    # Initialize smaller output array
    outputs = np.zeros((out_y, out_x))
    
    # Perform max pooling, iterate over the output coordinates (i, j)
    for i in range(out_y):
        for j in range(out_x):
            # Starting position in input image, "stride" takes place
            start_y = i * strides
            start_x = j * strides
            # Slice window from input, shape:(pool_size, pool_size)
            window = inputs[start_y : start_y + pool_size, start_x : start_x + pool_size]
            # take max val. in current window and assign to output
            outputs[i, j] = np.max(window)
    return outputs


def max_poolnn(inputs, pool_size=2, strides=1, verbose=True):
    ''' Max pooling implementation for a MaxPool2D layer of a neural network

    Parameters:
    -----------
    inputs: Input to be filtered. shape=(mini_batch_sz, n_chans, height img_y, width img_x)
        where n_chans is 1 for grayscale images and 3 for RGB color images
    pool_size: int. Pooling window extent in both x and y.
    strides: int. How many "pixels" in x and y to skip over between successive max pooling operations
    verbose: bool. I suggest making helpful print statements showing the shape of various things
        as you go. Only execute these print statements if verbose is True.

    Returns:
    -----------
    outputs: Input filtered with max pooling op. shape=(mini_batch_sz, n_chans, out_y, out_x)
        NOTE: out_y, out_x determined by the output shape formula. The input spatial dimensions are
        not preserved (unless pool_size=1...but what's the point of that?)

    What's new (vs max_pool):
    -----------
    - Multiple images (mini-batch support)
    - Images now have a color channel dimension too

    Hints:
    -----------
    - If you added additional nested loops, be careful when you reset your input image indices
    '''
    mini_batch_sz, n_chans, img_y, img_x = inputs.shape
    # Calculate output spatial diemnsions
    out_y = (img_y - pool_size) // strides + 1
    out_x = (img_x - pool_size) // strides + 1
    
    # intialize output as a 4D tensor
    outputs = np.zeros((mini_batch_sz, n_chans, out_y, out_x))
    
    # Spatial loops
    for i in range(out_y):
        for j in range(out_x):
            # calculating input coordinates based on stride
            start_y = i * strides 
            start_x = j * strides
            end_y = start_y + pool_size
            end_x = start_x + pool_size
            # Slicing across ALL batches & ALL channels
            window = inputs[:, :, start_y:end_y, start_x:end_x]
            # computing max over last 2 axes(Height and Width), result is (mini_batch_sz, n_chans)
            m_pool = np.max(window, axis=(2, 3))
            # results = ouput tensor
            outputs[:, :, i, j] = m_pool
    return outputs
