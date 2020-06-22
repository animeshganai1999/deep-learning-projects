import tensorflow as tf
import scipy.io
import numpy as np
import scipy.misc

#disable eager execution technique in version 2 of tensorflow
tf.compat.v1.disable_eager_execution()

class CONFIG:
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
    VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat' # Pick the VGG 19-layer model by from the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition".
    STYLE_IMAGE = 'style.jpg' # Style image to use.
    CONTENT_IMAGE = 'content.jpg' # Content image to use.
    OUTPUT_DIR = 'output/'

def load_vgg_model(path):
    '''
        0 conv1_1 ,shape : (3, 3, 3, 64)
        1 relu1_1 ,shape : ()
        2 conv1_2 ,shape : (3, 3, 64, 64)
        3 relu1_2 ,shape : ()
        4 pool1
        5 conv2_1 ,shape : (3, 3, 64, 128)
        6 relu2_1 ,shape : ()
        7 conv2_2 ,shape : (3, 3, 128, 128)
        8 relu2_2 ,shape : ()
        9 pool2
        10 conv3_1 ,shape : (3, 3, 128, 256)
        11 relu3_1 ,shape : ()
        12 conv3_2 ,shape : (3, 3, 256, 256)
        13 relu3_2 ,shape : ()
        14 conv3_3 ,shape : (3, 3, 256, 256)
        15 relu3_3 ,shape : ()
        16 conv3_4 ,shape : (3, 3, 256, 256)
        17 relu3_4 ,shape : ()
        18 pool3
        19 conv4_1 ,shape : (3, 3, 256, 512)
        20 relu4_1 ,shape : ()
        21 conv4_2 ,shape : (3, 3, 512, 512)
        22 relu4_2 ,shape : ()
        23 conv4_3 ,shape : (3, 3, 512, 512)
        24 relu4_3 ,shape : ()
        25 conv4_4 ,shape : (3, 3, 512, 512)
        26 relu4_4 ,shape : ()
        27 pool4
        28 conv5_1 ,shape : (3, 3, 512, 512)
        29 relu5_1 ,shape : ()
        30 conv5_2 ,shape : (3, 3, 512, 512)
        31 relu5_2 ,shape : ()
        32 conv5_3 ,shape : (3, 3, 512, 512)
        33 relu5_3 ,shape : ()
        34 conv5_4 ,shape : (3, 3, 512, 512)
        35 relu5_4 ,shape : ()
        36 pool5
        37 fc6 ,shape : (7, 7, 512, 4096)
        38 relu6 ,shape : ()
        39 fc7 ,shape : (1, 1, 4096, 4096)
        40 relu7 ,shape : ()
        41 fc8 ,shape : (1, 1, 4096, 1000)
        42 prob
    '''
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']
    
    def weights(layers,expected_layers_name):
        '''return weight and bias of a given layer'''
        wb = vgg_layers[0][layers][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layers][0][0][0][0]
        assert layer_name == expected_layers_name
        return W,b
    
    def relu(conv2d_layer):
        return tf.nn.relu(conv2d_layer)
    
    def conv2d(prev_layer,layers,layer_name):
        w,b = weights(layers = layers, expected_layers_name = layer_name)
        w = tf.constant(w)
        b = tf.constant(np.reshape(b,(b.size)))
        return tf.nn.conv2d(prev_layer,filters = w,strides = [1,1,1,1],padding = 'SAME')
    
    def conv2d_relu(prev_layer,layers,layer_name):
        return relu(conv2d(prev_layer, layers, layer_name))
    
    def avgpooling(prev_layer):
        return tf.nn.avg_pool(prev_layer,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
    
    # Constructs the graph model.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = avgpooling(graph['conv1_2'])
    graph['conv2_1']  = conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = avgpooling(graph['conv2_2'])
    graph['conv3_1']  = conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = avgpooling(graph['conv3_4'])
    graph['conv4_1']  = conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = avgpooling(graph['conv4_4'])
    graph['conv5_1']  = conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = avgpooling(graph['conv5_4'])
    
    return graph
    
def generate_noise_image(content_image,noise_ratio = CONFIG.NOISE_RATIO):
    noise_image = np.random.uniform(-20,20,(1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype('float32')
    input_image = noise_image*noise_ratio + content_image*(1-noise_ratio)
    return input_image

def reshape_image(image):
    image = np.expand_dims(image,axis = 0)
    image = image - CONFIG.MEANS
    return image

def save_image(path, image):
    
    # Un-normalize the image so that it looks good
    image = image + CONFIG.MEANS
    
    # Clip and Save the image
    image = np.clip(image[0], 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)
    
import matplotlib.pyplot as plt
from keras.preprocessing import image

def compute_content_cost(a_c,a_g):
    '''
    Parameters
    ----------
    a_c : 
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_g : 
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 

    Returns
    -------
    content cost

    '''
    m,n_h,n_w,n_c = a_g.get_shape().as_list()
    a_C_unrolled = tf.reshape(a_c, shape = [m, n_h,n_w, n_c])
    a_G_unrolled = tf.reshape(a_g, shape = [m, n_h,n_w, n_c])
    
    j_content = tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled))/(4*n_h,n_w,n_c)
    return j_content

def gram_matrix(A):
    ga = tf.linalg.matmul(A,A,transpose_b = True)
    return ga


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) 
    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))
    
    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    J_style_layer = 1 / (2*n_C*n_H*n_W)**2 * tf.reduce_sum((GS-GG)**2)
        
    return J_style_layer
    
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(model,STYLE_LAYERS):
    j_style = 0
    for layer_name,coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_s = sess.run(out)
        a_g = out
        j_style_layer = compute_layer_style_cost(a_s, a_g)
        j_style += coeff*j_style_layer
    return j_style

def total_cost(j_content,j_style,alpha = 10,beta = 40):
    j = alpha*j_content + beta*j_style
    return j

tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()

content_image = image.load_img('content.jpeg',target_size = (300,400))
content_image = reshape_image(content_image)
style_image = image.load_img('style.jpg',target_size = (300,400))
style_image = reshape_image(style_image)

generated_image = generate_noise_image(content_image)
plt.imshow(generated_image[0])

#LOad VGG-19 Model
model = load_vgg_model('imagenet-vgg-verydeep-19.mat')

# Assign the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_c = sess.run(out)
a_g = out
j_content = compute_content_cost(a_c, a_g)



# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))
# Compute the style cost
j_style = compute_style_cost(model, STYLE_LAYERS)

#Total cost
j = total_cost(j_content, j_style, alpha = 10, beta = 40)

optimizer = tf.compat.v1.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(j)

def model_nn(sess,input_image,iteraton = 200):
    sess.run(tf.compat.v1.global_variables_initializer())
    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(input_image))
    
    for i in range(iteraton):
        sess.run(train_step)
        generated_image = sess.run(model['input'])
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image

output = model_nn(sess, generated_image)



image = output+CONFIG.MEANS
plt.imshow(image[0])
plt.savefig('img.png')
    
    
