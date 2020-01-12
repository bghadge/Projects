M = 0.5;
m = 0.2;
I = 0.006;
g = 9.8;
b = 0.1;
l = 0.3;

q = (M+m)*(I+m*l^2)-(m*l)^2;
s = tf('s');

Kp=54.4;
Ki=190;
Kd=3.8;

Cart = tf([(I+m*l^2)/q 0 -g*m*l/q],[1 (b*(I+m*l^2))/q -((M+m)*m*g*l)/q -(b*m*g*l)/q 0]);
Pendulum = tf([m*l/q 0],[1 (b*(I+m*l^2))/q -((M+m)*m*g*l)/q -(b*m*g*l)/q]);
Controller = tf([Kd Kp Ki],[1 0]);
OpenLoop_pend = Controller*Pendulum;
OpenLoop_cart = Controller*Cart;
ClosedLoop_pend=feedback(OpenLoop_pend,1);
ClosedLoop_cart=feedback(1,OpenLoop_pend);

sim('Inverted_pendulum.slx',5);

figure(1)
step(Pendulum);
xlim([0 5]);
grid on;

figure(2)
step(Cart);
grid on;

figure(3)
step(ClosedLoop_pend);
xlim([0 5]);
grid on;

figure(4)
step(ClosedLoop_cart);
xlim([0 5]);
grid on;

figure(5)
rlocus(OpenLoop_pend);
xlim([0 5])
grid on

figure(6)
rlocus(Pendulum);
grid on

figure(7)
nyquist(OpenLoop_pend);

figure(8)
bode(OpenLoop_pend);

figure(9)
bode(ClosedLoop_pend);

