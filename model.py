from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPool2D, Dropout, concatenate, Conv2D, LeakyReLU, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError

def encoder_block(inputs, output_channel, conv_kernel=3, pool_kernel=2, lrelu_alpha=0.1):
    net = Conv2D(output_channel, conv_kernel, padding='same', kernel_initializer='he_normal')(inputs)
    conv = LeakyReLU(lrelu_alpha)(net)
    pool = MaxPool2D(pool_kernel)(conv)
    return conv, pool


def decoder_block(inputs, skip_conn_input, output_channel, conv_kernel=3, up_scale=2, lrelu_alpha=0.1):
    upsample = UpSampling2D(up_scale, interpolation='bilinear')(inputs)
    block_input = concatenate([upsample, skip_conn_input], axis=3)
    net = Conv2D(output_channel, conv_kernel)(block_input)
    net = LeakyReLU(lrelu_alpha)(net)
    return net


def unet(output_channel=3, pretrained_weights=None, input_size=(None, None, 4), first_kernel=7, second_kernel=5):
    inputs = Input(input_size)
    econv1, epool1 = encoder_block(inputs, 32, conv_kernel=first_kernel)
    econv2, epool2 = encoder_block(epool1, 64, conv_kernel=second_kernel)
    econv3, epool3 = encoder_block(epool2, 128)
    econv4, epool4 = encoder_block(epool3, 256)
    econv5, epool5 = encoder_block(epool4, 512)
    econv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(epool5)
    econv6 = LeakyReLU(0.1)(econv6)

    decoder_input = econv6
    net = decoder_block(decoder_input, econv5, 512)
    net = decoder_block(net, econv4, 256)
    net = decoder_block(net, econv3, 128)
    net = decoder_block(net, econv2, 64)
    net = decoder_block(net, econv1, 32)

    net = Conv2D(output_channel, 3, padding='same', kernel_initializer='he_normal')(net)
    
    model = Model(inputs = inputs, outputs = net)
    
    return model, econv6


if __name__ == "__main__":
    model, _ = unet()
    model.compile(optimizer = Adam(lr = 1e-4), loss=MeanAbsoluteError(), metrics=['accuracy'])

    print(model.summary())