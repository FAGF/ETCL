% 2015/4/8

function dx=model(t,x)
global unew dnew
x1=x(1);
x2=x(2);

 f=[-x(1)+x(2);-0.5*(x(1)+x(2))+0.5*x(2)*sin(x(1))^2];
  g=[0;sin(x(1))];
  k=[0;cos(x(1))];


dx=f+g*unew+k*dnew;
end