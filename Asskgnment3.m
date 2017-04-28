% @author Victor Wikén
% 2017-04-11
clear all;
clc;

%% initialization. 
addpath Datasets/cifar-10-batches-mat/;
[Xtrain,Ytrain,ytrain]= LoadBatch('data_batch_1.mat');
[Xval,Yval,yval]= LoadBatch('data_batch_2.mat');
%% for using all bathces as training
 [Xtest,Ytest,ytest]= LoadBatch('test_batch.mat');



%% REMOVE WHEN DOING LARGE TRAINIG!
mean_X = mean(Xtrain,2);
[Xtrain,Xval,Xtest] =  subtractMean(mean_X,Xtrain,Xval,Xtest); % Maybe not subtract from the training.
%% first initializations
[d,N] = size(Xtrain);
[K,~] = size(Ytrain);
stnDeviation = 0.001;
seed = 400;


m1 = 50;
m2= 30
m3 = 20
%% initialize parameters.

m(1)=m1;
m(2)=m2;
m(3) =m3

margin = 1.0000e-5;
%% store weights and bias in cell array
%
lambda =0;
numOfsamples=3;
[Weights,bias] = initializeParams(m,K,d,seed,stnDeviation);

[P,h,s] = EvaluateClassifier(Xtrain(:,1:numOfsamples), Weights,bias);
[grad_b,grad_W] = ComputeGradients(Xtrain(:,1:numOfsamples),Ytrain(:,1:numOfsamples),P,h,s,Weights,bias,lambda);
test = cell2mat(grad_b(2));
ComputeCost(Xtrain, Ytrain, Weights, bias, lambda)
% [grad_bnum, grad_Wnum] = ComputeGradsNum(Xtrain(:,1:numOfsamples), Ytrain(:,1:numOfsamples), Weights, bias, lambda, margin);
% [errorb,errorW,sumErrorb,sumErrorW]= displayError(grad_b,grad_bnum,grad_W, grad_Wnum,margin);


%% real deal

%% Training settings
 
% lambda = 0.001;%0.001 % training
% GDparams.n_batch = 128
% GDparams.n_epochs = 30;  % roughly around 5 to find good learning rate.
% GDparams.eta = 0.04; %% plottat 0.0467,
% 
% rho = 0.9 % set to 0.5,0.9, 0.99/ use 0.9 in training.
% decay_rate =0.95;  % works with 1 aswell
% GDparams.rho = rho;
% GDparams.decay_rate=decay_rate;
% 

%% How many datapoints used
% numOfsamplestrain = size(Xtrain,2);
% X.train =Xtrain(:,1:numOfsamplestrain);
% numOfsamplesval = 1000;
% X.val = Xval(:,1:numOfsamplesval);
% Y.train = Ytrain(:,1:numOfsamplestrain);
% Y.val  = Yval(:,1:numOfsamplesval);
% 
%% minibatch and display the accuracy 
% 
% [Wstar,bstar] = MiniBatchGD(X,Y,GDparams,Weights,bias,lambda);
% accuracy = ComputeAccuracy(Xtest,ytest,Wstar,bstar)

% %% display resulting images 
% for i=1:10
% im = reshape(Wstar(i, :), 32, 32, 3);
% sim{i} = (im - min(im(:))) / (max(im(:))- min(im(:)));
% sim{i} = permute(sim{i}, [2, 1, 3]);
% end
% 
% montage(sim, 'Size', [5,5]);



%% Functions 

%% Mini Batch algorithm.
function [Wstar,bstar] =  MiniBatchGD(X, Y, GDparams, W, b, lambda)

    % W1 mxd
    % W2 Kxm
    % b1 m × 1 
    % b2 K × 1 

% X and Y contains both training and validation data.
Xtrain = X.train;
Xval = X.val;
Ytrain = Y.train;
Yval = Y.val;

N = size(Xtrain,2);
n_epochs = GDparams.n_epochs;
n_batch = GDparams.n_batch;
eta = GDparams.eta;
rho = GDparams.rho;
decay_rate = GDparams.decay_rate;
% initialize the cost for training and validation in order to be able to
% plot as a function of epochs.
cost = zeros(1,n_epochs);
costval = zeros(1,n_epochs);
epochs = zeros(1,n_epochs);


k = size(W,2);
% for each epoch compute gradients for each minibatch.

%%initialize momentum params
vgradW = cell(1,k);
vgradb = cell(1,k);
for i=1:k
    vgradW(i) =	zeros(size(W(i)));
    vgradb(i)= zeros(size(b(i)));
    
end


for epoch = 1:n_epochs
    for j=1:N/n_batch
        %% Get the minibatches.
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = Xtrain(:,inds);    % indices from j_start to j_end.
        Ybatch = Ytrain(:, inds);
        
        %% evaluate the current W and b and compute the new gradients. 
        [P,h,s1] = EvaluateClassifier(Xbatch,W,b);
        
        [grad_btemp,grad_Wtemp] = ComputeGradients(Xbatch,Ybatch,P,h,s1,W,b,lambda);
         %% momentum
         for i = 1:k
         grad_Witemp = cell2mat(grad_Wtemp(i));
         grad_bitemp = cell2mat(grad_btemp(i));
         %%initialize....
         vgradW(i) = rho*vgradW(i) + eta*grad_Witemp;
         vgradb(i)= rho*vgradb(i) + eta*grad_bitemp;
         
          W(i) = cell2mat(W(i)) -vgradW(i);
          b(i) = cell2mat(b(i)) -vgradb(i);
         end
    end
    
    %save in order to plot the decreased loss on training and validation.
    cost(epoch) =  ComputeCost(Xtrain,Ytrain,W,b,lambda)
    epochs(epoch) = epoch;
    costval(epoch) = ComputeCost(Xval,Yval,W,b,lambda)
    eta = decay_rate*eta;
end

% len and lenmin1 is just n_epochs, n_epochs -1 respectively.
len = n_epochs+10;
lenmin1 = n_epochs-1;

%% Plotting the loss
plot(epochs,cost,epochs,costval);
title(['lambda = ' num2str(lambda) ', eta = ' num2str(eta)]);
legend('cost training ','cost validation');
xlabel('epoch') % x-axis label
ylabel('loss') % y-axis label

maxlim = max(max(costval,cost))+0.1;
minlim = min(min(cost,costval))-0.1;
% set limits for x- and y-axis
xlim([1,len]);
ylim([minlim,maxlim]); 
%set x-axis stepsize.
set(gca,'XTick',0:10:len); 

%% Return the new W and b.
Wstar = W;
bstar = b;
end
%% Compute Gradients
function [grad_b,grad_W] = ComputeGradients(X,Y,P,x,s,W,b,lambda)
%X:dxn
%Y: Kxn
%P: Kxn
%W: Kxd
%grad_W: grad matrix of the cost J relative to W, KxD
%grad_b: grad vector of the cost J relative to b, Kx1

     % W1 mxd
    % W2 Kxm
    % b1 m × 1 
    % b2 K × 1 

k = size(W,1)
    
gradJ_WAvg = cell(k,1);
gradJ_bAvg = cell(k,1);
gradJ_W = cell(k,1);
gradJ_b = cell(k,1);

[K,N] = size(Y);


for i = 1:k
    display("hello elr")
%% initialize b NECCESARY?
sizebi = size(cell2mat(b(i)))
sizeWi = size(cell2mat(W(i)))
gradJ_b(i)= {zeros(sizebi)};% står d i slides but must be wrong 
gradJ_W(i) = {zeros(sizeWi)};
storedsizebi =size(cell2mat(gradJ_b(i)))
storedsizeWi =size(cell2mat(gradJ_W(i)))
end
sizeBgrad = gradJ_b
sizeWgrad = gradJ_W
komigennUb1= size(cell2mat(gradJ_b(1)))
komigennUb2 =size(cell2mat(gradJ_b(2)))

% for every picture calculate calculate gradients, sum gradients for b and
% W, in order to take avarage.
for i = 1 : N
    Xbat = X(:,i); 
    YT= Y(:,i)';   
    Ptemp = P(:,i); 
    size(YT)
    size(Ptemp)% probabilities for the classes for one image per loop iteration 
    YTP = YT*Ptemp; % correct  dim: 1x1
    Pdiag = diag(Ptemp);
    PPT = Ptemp*Ptemp';
    PgradJ_p = -(YT/YTP);
    
    %step 1
    g = PgradJ_p*(Pdiag - PPT);
    
    %%for each layer k.
    % step 2
  %  k
    %display("hello")
    for j = k:-1:1 
    %%isplay("innie i for")
    gradJbitemp = cell2mat(gradJ_b(j));
    gradJbiSize= size(gradJbitemp)
    gisizebeforbi = size(g)
    gradJbitemp = gradJbitemp + g';
    komigennugradBj = size(gradJbitemp)
    gradJ_b(j) = {gradJbitemp};
    %% tror det har makes sense, lsm för varje bild, do every h for varje W
    %% Kan vara fel har med x(j-1) men andrade fran 0-k-1 to 1-k
    h = cell2mat(x(j));
    %h
    %%knas..
    htemp = h(:,i);
    %size(g)
    %sizehtemp = size(htemp)
   ghtemp =  g'*htemp';
    gradJ_wi = cell2mat(gradJ_W(j));
    sizegradJWi = size(gradJ_wi)
    gradJ_W(j) = {gradJ_wi +  ghtemp + 2*lambda*cell2mat(W(j))}; % g': Kx1 h:mxN => h(:,i)' = 1xm, => gJW2 = Kxm
    %step 3
        if(j>1)
        g = g*cell2mat(W(j));
        % g = 1xK* Kxm => 1xm
        %% should it be s(j-1) here check 
        si = cell2mat(s(j-1));
        %%si = cell2mat(s(j)); version 2 s(j) not s(j-1)
        sjtemp = si(:,i); %mx1
    %% INDICATOR FUNCTION IND 1 if true, 0 if false.
        sjtemp(sjtemp>0) =1;
        sjtemp(sjtemp<0) =0;
        g = g*diag(sjtemp);
        end
    end
%     %% gets me to think max(0,s1) is not correct
%     %step 4
%     gradJ_b1 = gradJ_b1 + g;
%     gradJ_W1 = gradJ_W1 + g'*Xbat';


end

%% should probably be removed?
for i = 1:k
% avarge gradients for W  and b, and add regularization term to gradients for W.
gradJ_WAvg(i) = {cell2mat(gradJ_W(i))./N} 
gradJ_bAvg(i) = {cell2mat(gradJ_b(i))./N}   
 
end

%% return as cell?

grad_b = gradJ_bAvg;
grad_W = gradJ_WAvg;

end


function [errorb,errorW,sumerrorb,sumerrorW]= displayError(grad_b,grad_bnum,grad_W, grad_Wnum,margin)

k = size(grad_W,1)
% kinError = k
errorb= cell(1,k);
errorW = cell(1,k);
for i = 1:k-1
    tb = cell2mat(grad_b(i));
%     tbsize = size(tb)
    tw = cell2mat(grad_W(i));
%     twsize = size(tw)
    tbnum = cell2mat(grad_bnum(i));
    twnum = cell2mat(grad_Wnum(i));
    errorb(i) = {ComputeError(tb,tbnum,margin)};
    errorW(i) = {ComputeError(tw,twnum,margin)};
end
errorW(k) = {ComputeError(cell2mat(grad_W(k)), cell2mat(grad_Wnum(k)),margin)};
errorb(k) = {cell2mat(grad_b(k))- cell2mat(grad_bnum(k))};

%% they are more or less the same.
sumerrorb = zeros(1,k);
sumerrorW = zeros(1,k);

for i = 1:k
    errbi = cell2mat(errorb(i));
    errWi = cell2mat(errorW(i));
%     sizeofstoredCostbi =size(errbi)
%     sizeofstoredCostWi =size(errWi)
    sumerrorb(1,i) = sum(sum(errbi))
    sumerrorW(1,i) = sum(sum(errWi))
end



end


function error = ComputeError(A,B,margin)

    error = abs(A-B)/max(margin,abs(A)-abs(B));


end




%% Compute accuracy 
function acc = ComputeAccuracy(X,y,W,b)

    p = EvaluateClassifier(X,W,b);
    [~, argmaxLabel] = max(p); %take the class with highest probabilites for each image
    res = int8(argmaxLabel')-y; % the differance between the correct klasses and the "guessed" classes
    correctVec = find(res==0); % where the differance is zero we have a correct classified image
                               % all none-zero are incorrect.
    totalCorrect  = length(correctVec); %number of correctly classified images
    len = length(res);                  %number of images
    acc = totalCorrect/len;      %the 
    


end

%% Evaluate Classifier and softmax

% EvaluateClassifier
function [P,x,si] = EvaluateClassifier(X, W,b)

    % W1 mxd
    % W2 Kxm
    % b1 m × 1 
    % b2 K × 1
     %% should this really be k or should it be size(W,1) +1
    %% borja har ngt knas
    k= size(W,1); %% why is k=1?
    si = cell(k,1);
    x = cell(k,1);
    % if i = 0;...
    x(1) = {X};
    %%kis = k
    for i = 1:k-1
        Wi = cell2mat(W(i));
        xi = cell2mat(x(i));
        bi = cell2mat(b(i));
        sitemp = Wi*xi + bi;
        %sizesitemp = size(sitemp)
        si(i) = {sitemp};
        x(i+1) ={max(0,sitemp)};
%         size_h = max(0,sitemp)
    end
    %think about this, should it not be included? menaing only as variable
    %s
    Wk = cell2mat(W(k));
    xk = cell2mat(x(k));
    bk = cell2mat(b(k));
%     Wksize = size(Wk)
%     xksize = size(xk)
%     bksize =size(bk)
    si_k = Wk*xk + bk;
    %% si_k should be KxN dvs  ex 10x3
    si(k) = {si_k};
%     sizesi_k = size(cell2mat(si(k)))

    P= softMax(si(k));
%     P_size = size(P)
end


% softmax
function P = softMax(s)
    %%think this is needed
    s= cell2mat(s);
    %size_s_in_Softmax = size(s);
    [srow,~] = size(s);
    onesVec = ones(srow,srow); %10x10, s= 10x100   = > dim: 10x100
    f= exp(s);
    nom = onesVec*f;        %dim:10x100   sum over exp(s)
    P = f./nom;        

end



%% ComputeCost
function J = ComputeCost(X, Y, W, b, lambda)
    %W contains both W1 and W2
    %b contains b1 and b2
    k =size(W,2);
    
    [p,h] = EvaluateClassifier(X,W,b);
    % KxN.*kxN => dim:kxN for every column of the matrix 
    %every element is zero except for the class the image belongs to
    lc = -log(Y.*p);                
    [~,setsize] = size(X);   
    % gets the loss for every image
    %(since every element is zero except the correct class we get the loss for that class)
    [val,~] = min(lc);   
    %avaraged of all the images.
    preterm= sum(val)/setsize;
    postterm = 0;
    
    %% post term, with k could be wrong..think its correct.
    for i = 1:k
    postterm = postterm +  lambda*sum(sum(cell2mat(W(i)).^2));
    end
    J = preterm+postterm;   
end


%% Other used functions

%% initialize Parameters

function [W,b] = initializeParams(m,K,d,seed,stnd)
% W1 mxd
% W2 Kxm
% b1 m × 1 
% b2 K × 1
%m = array [50,30,...20] each value is number of nodes for the ith:layer
rng(seed);
k = size(m,2)+1
W = cell(k,1)
b = cell(k,1)
prevm= zeros(k,1)
prevm(1) = d;

for i = 1: k-1
    W(i) = {stnd*randn(m(i),prevm(i))}; %mxd
    prevm(i+1) = m(i)
    b(i) = {stnd*randn(m(i),1)}; % mx1
 
    
end
    
W(k) = {stnd*randn(K,prevm(k))}; %Kxm
b(k) = {stnd*randn(K,1)}; % Kx1
display("hej")
W
b
end

%% Load Batch
function [X,Y,y] = LoadBatch(filename)
   A = load(filename);
   X = A.data';
   X= im2double(X); 
   [~,cols]=  size(X);
   temp= zeros(10,cols);
   
   for i =1:cols
         a = int8(A.labels(i))+1;   % + 1 to label 1-10 instead of 0-9
         temp(a,i) = 1;
   end
   Y=temp;
   y= int8(A.labels)+1;   %+1 to get index 1-10 
    
end

function[X1,X2,X3] =  subtractMean(mean,X1,X2,X3)
    X1 = X1 - repmat(mean,[1, size(X1, 2)]);
    X2 = X2 - repmat(mean,[1, size(X2, 2)]);
    X3 = X3 - repmat(mean,[1, size(X3, 2)]);
end

%% functions to compute gradients numerically .

function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

c = ComputeCost(X, Y, W, b, lambda);
%bh = waitbar(0, 'bias');

for j=1:length(b)

    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
      %  waitbar(i/length(b{j}), bh)

        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b{j}(i) = (c2-c) / h;
    end
end
wh = waitbar(0, 'Weights');
for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})   
       waitbar(i/numel(W{j}), wh)  

        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W{j}(i) = (c2-c) / h;
    end
end
end


function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end