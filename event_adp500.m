% 2015/4/8


clc
clear
close all;

%-----------------initialing-----------------------------------
global unew dnew
a=1;
b=0.1;            % learning rate
c=0.1;

R=1;
gamma=3;          
Q=eye(2);
belta=0.2;
L=5;

x0=[1 -1];    % initial state
x_next=x0;
X=[x0];
hat_x=x0;

hiden_neu=5;

wcn=0.5*zeros(hiden_neu,1);
%wcn=[0.327870349578293,0.0178558392870948,0.424564652934389,0.466996623878775,0.339367577428887]';
Wcn=[wcn'];


wan=zeros(hiden_neu,1);
% wan=[0.757740130578333,0.743132468124916,0.392227019534168,0.655477890177557,0.171186687811562]';
Wan=[wan'];

wwn=zeros(hiden_neu,1);
Wwn=[wwn'];

t0=0.1;
K=3500;
U=[];
D=[];
S=[]; 
E=[];
F=[];
V=[];

sum1=0;
nbg1=1;
Xbg1=zeros(5,10);
Ybg=zeros(5,10);
N=zeros(1,5);

for i=1:K
    i
    x=x_next;
    f=[-x(1)+x(2);-0.5*(x(1)+x(2))+0.5*x(2)*sin(x(1))^2];
    g=[0;sin(x(1))];
    hat_g=[0;sin(hat_x(1))];
    k=[0;cos(x(1))];
    
    phix=[x(1)^2 x(1)*x(2) x(2)^2 x(1)^4 x(2)^4]';     % ¼¤Àøº¯Êý  fai
    dphix=[2*x(1) 0; x(2) x(1);0  2*x(2); 4*x(1)^3 0; 0 4*x(2)^3];
    dphix_u=[2*hat_x(1) 0; hat_x(2) hat_x(1);0  2*hat_x(2); 4*hat_x(1)^3 0; 0 4*hat_x(2)^3];
    
      d=0.5*(1/(gamma^2))*k'*dphix'*wwn;
      u=-0.5*inv(R)*hat_g'*dphix_u'*wan; 
     if i<3000
         t=i/10;
         unew=((u)+exp(-0.006*t)*1.5*(sin(t)^2*cos(t)+sin(2*t)^2*cos(0.1*t)+sin(-1.2*t)^2*cos(0.5*t)+sin(t)^5+sin(1.12*t)^2+cos(2.4*t)*sin(2.4*t)^3));
         dnew=((d)+exp(-0.006*t)*1.5*(sin(t)^2*cos(t)+sin(2*t)^2*cos(0.1*t)+sin(-1.2*t)^2*cos(0.5*t)+sin(t)^5+sin(1.12*t)^2+cos(2.4*t)*sin(2.4*t)^3));
     else
         unew=u;
         dnew=d;
     end
     
   tspan=(i-1)*t0:t0:i*t0;
   [t,xx]=ode23('model',tspan,x);
    x_next=xx(end,:); X=[X;x_next];
   
    s=dphix*(f+g*u+k*d);                                   %fai_=dfai*(f+gu+kd)
    Y=(-[x(1) x(2)]*Q*[x(1) x(2)]'-u*R*u'+(gamma^2)*d*d');             % reward
    e=wcn'*s-Y;  
       m=s/(1+s'*s)^2; 
        oldrank1=rank(Xbg1*Xbg1');   
    if norm(s-Xbg1(:,nbg1))/norm(s)>0.2
         nbg1=nbg1+1;
         Xbg1(:,nbg1)=s;
         Ybg(:,nbg1)=m;
         N(nbg1)=Y;
      if oldrank1==5&&nbg1>10
           nbg1=1;
        end  
    end
   sum1=0;
   if i<K
   for kk1=2:nbg1
    sum1=sum1+Ybg(:,kk1)*(Xbg1(:,kk1)'*wcn-N(kk1));
   end
   else
       sum1=0;
   end
    
    
    
    
   % wcn_next=wcn-a*(s./(s'*s+1)^2)*e';
   wcn_next=wcn-a*(s./(s'*s+1)^2)*e' -0.3*a*sum1;
 
 wwn_next=wcn_next;
 e_T=(1-belta^2)/L^2*(norm(x))^2+1/L^2*(norm(u))^2-gamma^2/L^2*norm(d)^2;
  S=[S;e_T];
 e=hat_x-x;
 e_d=norm(e);
  flag=0;
  if norm(e)^2>e_T
   flag=1; 
%   e_d=sqrt(e_T);
   D1=dphix_u*hat_g*inv(R)*g'*dphix_u';
 %  wan_next=wan-b*(F2*wan-F2*wcn)+0.25*b*D1*wan*(s'./(s'*s+1)^2)*wcn;
    wan_next=wcn_next;
   hat_x=x_next;
  else
     hat_x=hat_x;
     wan_next=wan;    
  end
 
  
 wcn=wcn_next;
 wan=wan_next;
 wwn=wwn_next;
 Wcn=[Wcn;wcn'];
 Wan=[Wan;wan'];
 Wwn=[Wwn;wwn'];
 U=[U;u];
 D=[D;d];
 E=[E;e_d];
 F=[F;sqrt(e_T)];
 V=[V;flag];
end
 
    t1=0:t0:K/10;

figure(1)
plot(t1,Wcn,'LineWidth',2);
grid on;

figure(2)
plot(t1,Wan,'LineWidth',2);
grid on;
figure(3)
plot(t1,Wwn,'LineWidth',2);
grid on;   
    
figure(4)
plot(t1,X);
grid on; 

t2=0.1:t0:K/10;
% 
% figure(5)
% plot(t2,U);
% grid on;
% figure(6)
% plot(t2,D);
% grid on;   
    
 figure(7)
plot(t2,E,'r')
hold on;
plot(t2,F)
grid on;   
       
  I=[];
for i=1:K
    if V(i)==1
        I=[I;i];
    end
    i=i+1;
end

l=length(I);
T=[0.01*I(1)];
for i=1:l-1
    t1=0.1*(I(i+1)-I(i));
    T=[T;t1];
    i=i+1;
end
figure(8)
plot(0.1*I,T,'*')   