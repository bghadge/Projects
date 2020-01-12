%% To test each sub-problem, use Run Section!

%% P3.1
% test Sigmoid Function
x1 = rand(10,4);
y = sigmoid(x1, 0);
y_ref = sigmoid_ref(x1, 0);
% test Derivative of Sigmoid Function
x2 = rand(6,2);
dy = sigmoid(x2, 1);
dy_ref = sigmoid_ref(x2, 1);

%% P3.2
labels = randi([0 9],100,1);
num_classes = 10;
labels_oh = one_hot_encode(labels, num_classes);
labels_oh_ref = one_hot_encode_ref(labels, num_classes);

%% P3.3
current_input = rand(576,1);
current_labels = rand(10,1);
num_hidden_neurons = randi([10 100],1);

%% Write your code here:
%% if you just want to test this function, you need to figure out the dimension of weight here
% W1 = rand(% put correct dimension here%);
% W2 = rand(% put correct dimension here%);
% b1=rand(% put correct dimension here%);
% b2=rand(% put correct dimension here%);
%
% [l1, a1, l2, a2, loss] = simple_nn_fwd(current_input, current_labels, W1, W2, b1, b2)

[l1_ref, a1_ref, l2_ref, a2_ref, loss_ref] = simple_nn_fwd_ref(current_input, current_labels, W1, W2, b1, b2);

%% P3.4
current_input = rand(576,1);
current_labels = rand(10,1);
num_hidden_neurons = randi([10 100],1);
%% Write your code here:
%% if you just want to test this function, you need to figure out the dimension of weight here
% W1 = rand(% put correct dimension here%);
% W2 = rand(% put correct dimension here%);
% b1 = rand(% put correct dimension here%);
% b2 = rand(% put correct dimension here%);
% l1= rand(% put correct dimension here%);
% l2 = rand(% put correct dimension here%);
% a1=sigmoid(l1,0);
% a2=exp(l2);
% loss = rand(1,1);
% lr=rand(1,1);
% [updated_w1,updated_w2,updated_b1,updated_b2] = backprop_simple(...
%     current_input, current_labels, loss,...
%     l1, a1, l2, a2, W1, W2, b1, b2, lr);

[updated_w1_ref,updated_w2_ref,updated_b1_ref,updated_b2_ref] = backprop_simple_ref(...
    current_input, current_labels, loss,...
    l1, a1, l2, a2, W1, W2, b1, b2, lr);

%% P3.5
[W1, W2, b1, b2] = run_nn('mnist.mat', 10, 0);
[W1_ref, W2_ref, b1_ref, b2_ref] = run_nn_ref('mnist.mat', 10, 0);

