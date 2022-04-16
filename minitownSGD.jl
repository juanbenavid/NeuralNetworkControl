
# Using gradient descent to derive the minitown system equations
# Juan Benavides 4/16/2022

using Plots
using Flux

include("MINITOWNDATA.jl")
dat = minitownData()

tankFlow = dat[1]
tankHead = dat[2]
pump1 = dat[3]
pump2 = dat[4]
pumpsFlow = dat[5]
demandFrac = dat[6]

#plot(range(0,170,step=1),pumpsFlow)
#plot(range(0,170,step=1),demandFrac)

plot(range(0,170,step=1),tankFlow)

# construct a training set
# each X is a vector of state vars we care about [pump1, pump2, tank head , demand]
# each Y is the corresponding tank flow for that state

X_Train = [[p1, p2, tH, d] for (p1,p2,tH,d) = zip(pump1,pump2,tankHead,demandFrac)]
Y_train = [tF for tF= tankFlow ]

# Assume that a linear equation of this form is enough to describe the system:
# tankflow = A*p1 + B*p2 + C*tH +D*d + E

# We can declare some global vars for the coefficients
# these are the parameters that we will train with gradient descent
global A = randn(1)[1]
global B = randn(1)[1]
global C = randn(1)[1]
global D = randn(1)[1]
global E = randn(1)[1]

# fhat is our function "approximator" 
fhat(a,b,c,d,e,p1,p2,tH,dem) = a*p1 + b*p2 + c*tH +d*dem + e;

# our loss function will be the mse
loss(a,b,c,d,e,p1,p2,tH,dem,target) = (target - fhat(a,b,c,d,e,p1,p2,tH,dem))^2;

# Flux easily gives us the gradient of the loss function wrt to each variable in this form:
∇_A = gradient(loss,A,B,C,D,E,1.0,1.0,74.5,0.92,-34.56)[1]

# if we adjust A against this gradient we can slowly reduce our loss function
# we'll use a small learning rate to make sure we converge
# 1 epoch means we have iterated through each X in our training set
# we'll run 30k epochs and see what happens

learning_rate = 0.00001

for epoch = 1:30000
    _loss = 0
    for (X,Y) = zip(X_Train,Y_train)
        ∇_A = gradient(loss,A,B,C,D,E,X[1],X[2],X[3],X[4],Y)[1]
        ∇_B = gradient(loss,A,B,C,D,E,X[1],X[2],X[3],X[4],Y)[2]
        ∇_C = gradient(loss,A,B,C,D,E,X[1],X[2],X[3],X[4],Y)[3]
        ∇_D = gradient(loss,A,B,C,D,E,X[1],X[2],X[3],X[4],Y)[4]
        ∇_E = gradient(loss,A,B,C,D,E,X[1],X[2],X[3],X[4],Y)[5]

        global A -= learning_rate * ∇_A
        global B -= learning_rate * ∇_B
        global C -= learning_rate * ∇_C
        global D -= learning_rate * ∇_D
        global E -= learning_rate * ∇_E

        
        _loss += loss(A,B,C,D,E,X[1],X[2],X[3],X[4],Y)
    end
    if epoch %1000 == 0
        println(_loss)
    end
    
end

# lets grab our predictions
preds = [fhat(A,B,C,D,E,X[1],X[2],X[3],X[4]) for X=X_Train]

# matches up quite well!
plot(range(0,170,step=1),Y_train)
plot!(range(0,170,step=1),preds)
