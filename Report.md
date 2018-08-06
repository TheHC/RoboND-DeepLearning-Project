## Project: Follow Me

---



[//]: # (Image References)

[image1]: ./arch.png



## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

 

---
### General architecture :

 

![alt text][image1] 

After many experiments, I've settled on this architecture. It uses 3 encoders and thus 3 decoders with 3 skip connections 
and of course a 1x1 convolution. 

The implementation is done in the code as follows : 

    def fcn_model(inputs, num_classes):
    
        #  Encoder Blocks.     
        enc=encoder_block(inputs, 32, 2)
        enc2=encoder_block(enc, 64, 2)
        enc3=encoder_block(enc2, 128, 2)
    
        # 1x1 Convolution layer using conv2d_batchnorm().
        conv_1x1=conv2d_batchnorm(enc3, 256, kernel_size=1, strides=1)
        
        # Decoder Blocks as the number of Encoder Blocks
        dec1=decoder_block(conv_1x1, enc2, 128)
        dec2=decoder_block(dec1, enc, 64)
        x=decoder_block(dec2, inputs, 32)
        
        # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
        return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)

In the following sections, I will talk about the architecture of each block and the role of each layer.

#### 1. Encoders

Each encoder contains  : 

1 - A separable Convolution layer : 

As mentioned in the lessons,separable convolutions,
also known as depthwise separable convolutions,
comprise of a convolution performed over each channel of an input layer
and as opposed to a regular convolution, reduces the number of parameters. 

2 - Batch Normalization : 

By normalizing each output of the encoder we ensure that each input to a layer is normalized.
And normalization can have many benefits for the model : training faster, 
allowing higher learning rates and regularization. 

The encoder bloc is implemented in the code as follows : 

    def encoder_block(input_layer, filters, strides):
        
        # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
        output_layer=separable_conv2d_batchnorm(input_layer,filters,strides)
        return output_layer


The first encoder captures simple image features and each time we add an encoder 
the features captured are more complex.

#### 2. 1x1 convolution 

In a classical network we use flat fully connected layer after the convolution layers 
to prepare for the output. By doing that we lose the spacial information and we won't have 
a pixelwise semantic segmentation. This why the FCN is more efficient in a scene understanding.
By replacing the flat fully connected layer with the  1x1 convolution we can keep the spacial information
and also make the model deeper without adding a lot of parameters. 

The 1x1 convolution is implemented as follows :  

    # 1x1 Convolution layer using conv2d_batchnorm().
    conv_1x1=conv2d_batchnorm(enc3, 256, kernel_size=1, strides=1) 
 

#### 3. Decoders

After that the encoders extracted features from the image, the role of decoders is 
mainly to upscale the result to the size of the input image. This way we have a prediction per pixel. 

Each decoder contains : 

1- Upsampling layer 

As it name suggests, this layer helps upsample the previous one to a desired resolution.
In the lessons two upsampling techniques have been discussed: transposed convolution and bilinear upsampling.
As opposed to transposed convolutions the bilinear upsampling does not add parameters to the model and thus does not learn but it 
helps get better learning and predicting speed considering hardware performance. 

2- Concatenation 

This is where we use skip connections by concatenating an output of a previous layer to the new bloc.
As discussed earlier, each time we go deeper, the features captured become more complexe. At the end the prediction 
is done only using these features and maybe  precious informations was lost in some more simple features predicted in some previous layers. 
We can gain more informations by joining some simple features to the complexe ones by using skip connections.

3- 2 Separable convolution layers 

The separable layers used in the decoders work the same as those in the encoders blocks and have the same benefits as discussed.

The decoder blocs are implemented as follows : 

    def decoder_block(small_ip_layer, large_ip_layer, filters):
        
        # TODO Upsample the small input layer using the bilinear_upsample() function.
        
        output_layer=bilinear_upsample(small_ip_layer)
        
        # TODO Concatenate the upsampled and large input layers using layers.concatenate
        
        output_layer=layers.concatenate([output_layer, large_ip_layer])
        
        # TODO Add some number of separable convolution layers
        
        output_layer=separable_conv2d_batchnorm(output_layer,filters,1)
        output_layer=separable_conv2d_batchnorm(output_layer,filters,1)
        
      
    
    return output_layer 


#### 4. Parameters

After going through the lessons and labs, I had a decent understanding of the role 
each parameter played in the  performance of the model training. 

Tuning these parameters is a crucial step and is as important as the selection of the 
network architecture. 

For this project I chose to use my NVIDIA GPU ( geforce 1060 6go) and CUDA. So besides the 
model performance, there was the learning time to consider. The final model took aroung 9 hours 
to complete the training. 

Generally speaking , I think the best way to tune the parameters is to begin with a low learning rate
and a high number of epochs. By following the accuracy graphes we can get a better idea on when the performance 
platoons. 

At first I chose a learning rate of 0.001 and 50 epochs, after that I lowered the learning to 0.0001
and increased the epochs to 250. This showed me that the learning platoons around 50 epochs and reaches a better score using the late value. 

The bach size I chose was 25 because beyond that I had Hardware issues due to limited ressources.


I guess lowering the learning rate is the most important step in tuning the parameters and the rest depends on 
materiel and learning database. 

#### 5. Limitations 

My model is not deep enough to be used to work with a wide variety of objects. I guess 3 convolution layers in encoding 
don't have the depth required to capture very complex features. For example if we were to chose to follow dogs and not cats I don't think my model 
will do a great job since differentiating between these 4 legged animals would required analysing some specific features like the fur, size of the head, size of the tail, etc.

Also, my model contains only one decoder. Maybe training many different decoders on 
  different detection tasks would help making it better at scene understanding as mentioned in the lessons.
And this is very important if we are to follow a car for example. 

I trained my model only on the data provided as I was tight on time. But I believe that 
capturing a good learning database is crucial. Even after this project, I ll try to experiment different 
databases and capture different images. The idea is to have good variety. we can think of it as sampling 
the reality the model will encounter. The sample should be representative and not give importance to some situations or features over others. 
So I ll try take pictures of different hero poses, in different places, with different crowds and at different drone altitudes.



   
