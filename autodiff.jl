# Experimenting with autodifferentiation
# in Julia with Flux
# Juan Benavides 12/4/2021

using Flux
using Plots

# We can see how this ties back to 
# the training we do in neural networks
# lets say we want to train
# parameters m and b in y= mx+b to
# approximate the function y = 3x + 5

# Initilize some random params
M = randn(1)[1]
B = randn(1)[1]
X = randn(1)[1]

# our fhat function approximates y = 3x+5
fhat(m,b,x) = m*x + b;

# MSE loss to measure our performance
loss(m,b,x) = ((3x+5) - fhat(m,b,x))^2;


# for example, loss with current param values
loss(M,B,X)

# We find the gradient wrt to M and B 
∇_m = gradient(loss,M,B,10)[1]
∇_b = gradient(loss,M,B,10)[2]

# we adjust M and B against the gradient
# multiplying by a learning rate (0.01)
M -= 0.01 * ∇_m
B -= 0.01 * ∇_b

# our loss is now lower
loss(M,B,X)

# if we do this continously
# we can train our params down
# until we are close enough
for epoch = 1:10000
    ∇_m = gradient(loss,M,B,X)[1]
    ∇_b = gradient(loss,M,B,X)[2]
    X = randn(1)[1]
    M -= 0.001 * ∇_m
    B -= 0.001 * ∇_b  
    if epoch %1000 == 0
        println(loss(M,B,X))
    end
end

#We got pretty close

M
B

fhat(M,B,10)

3(10) + 5

# in neural networks
# we do this same kind of thing
# but the networks keep track of thousands of params
# that get updated at each epoch, with more complicated loss functions
# and multiple layers, some that can be complex functions
# such as convultional layers, activations etc.