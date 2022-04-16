using Flux
using Flux: Data.DataLoader
using Flux: onehotbatch, onecold, crossentropy
using Flux: @epochs
using Statistics
using MLDatasets
using Plots

# Load the data
# our images come in a 28x28x60000 3d array
# our labels come in a 60000 x 1 vector
train_x, train_y = MNIST.traindata(Float32)
test_x, test_y = MNIST.testdata(Float32)

heatmap(train_x[:,:,1]) # this is a 28x28 matrix/img

train_y[1]  # this is just an int

# Add the channel layer
train_x = reshape(train_x, 28, 28, 1, :)
test_x = reshape(test_x, 28, 28, 1, :)

train_x[:,:,:,1] # 28x28x1 matrix/1 channel img

heatmap(train_x[:,:,1,1]) # same image

# Encode labels
train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)

train_y[:,1] # this is a 10 element one hot vector

# one cold just reverses a onehot vector and returns the corresponding label
# we do this because our model will output 10 element vector 
# with the magnitude of each element being the likelyhood of that label 0:9
onecold(train_y[:,1],0:9)

# the dataloader function creates an iterable
# object that we can loop through when we train
# it also creates our minibatches and shuffles them
# at each epoch
DATA = DataLoader((train_x, train_y), batchsize=128, shuffle=true)

# just to see what dataloader is doing
count = 0
for (ex, yi) in DATA
    if count % 1000 == 0
        println(size(yi))
        println(size(ex))
    end
    count +=1
end

# The chain function lets us specify our
# netowrk architecture, this is analogous to the nn.sequential in pytorch
# or my fhat function in base julia

net = Chain( flatten,
  Dense(28*28, 512,relu),
  Dense(512, 512,relu),
  Dense(512, 10),
  )


# we can define a loss function from one of the many
# that come built in with Flux 
# cross entropy is the most appropriate
# for classification networks  

loss(x, y) = Flux.Losses.logitcrossentropy(net(x), y)

# descent is flux's built in SGD, the argument we pass is the learning rate
optimizer = Descent(0.003)

# Flux.params is an easy way to get all of the parameters
# that we want to track and update while we train
ps = Flux.params(net)

# Flux.train does all of the work for us 
#implements SGD, calculates gradients and updates parameters
number_epochs = 5
@epochs number_epochs Flux.train!(loss, ps, DATA, optimizer)

onecold(softmax(net(train_x[:,:,:,1])),0:9)[1] == onecold(train_y[:,1],0:9)
# we can check our accuracy on both the train and test sets
correct = 0
for i = 1:length(train_y[1,:])
    if onecold(softmax(net(train_x[:,:,:,i])),0:9)[1] == onecold(train_y[:,i],0:9)
        correct += 1
    end
end
correct* 100/length(train_y[1,:])

correct = 0
for i = 1:length(test_y[1,:])
    if onecold(softmax(net(test_x[:,:,:,i])),0:9)[1] == onecold(test_y[:,i],0:9)
        correct += 1
    end
end
correct* 100/length(test_y[1,:])


