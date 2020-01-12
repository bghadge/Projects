clear all, close all, clc

m = 1;
M = 5;
L = 2;
g = -10;
d = 1;

A = [0 1 0 0;
    0 -d/M -m*g/M 0;
    0 0 0 1;
    0 -d/(M*L) -(m+M)*g/(M*L) 0];



B = [0; 1/M; 0; 1/(M*L)];
eig(A)

Q = [1 0 0 0;
    0 1 0 0;
    0 0 10 0;
    0 0 0 100];
R = .0001;

%%
det(ctrb(A,B))

%%
K = lqr(A,B,Q,R);

tspan = 0:.001:10;

y0 = [-3; 0; pi+.1; 0];
[t,y] = ode45(@(t,y)model(y,m,M,L,g,d,-K*(y-[1; 0; pi; 0])),tspan,y0);

figure('units','normalized','outerposition',[0 0 1 1]);

for k=1:100:length(t)
    animate(y(k,:),m,M,L);
end
