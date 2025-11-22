% Adaboost
clear
clc

%初始化设置
iter=3;
y=[1,1,1,-1,-1,-1,1,1,1,-1]';
x=[1,2,3,4,5,6,7,8,9,10]';
theta=x-0.5;
p=[-1,1]';
D=ones(10,1)*(1/10);
dec=0;

for i=1:1:iter
    [p_pred,theta_pred,loss_pred]=find_min(D,x,y,p,theta)  %打印参数
    alpha=0.5*log((1-loss_pred)/loss_pred);
    h=p_pred.*sign(x-theta_pred);
    Z=0;
    for k=1:1:10
        Z=Z+D(i,1)*exp(-alpha*y(i,1)*h(i,1));
    end
    D=(D.*exp(-alpha.*y.*h))./(Z);
    dec=dec+alpha.*h;
end

H=sign(dec);
res=H-y;

function loss= error(D,x,y,p,theta)
loss=0;
for i=1:1:10
    h=p*sign(x(i,1)-theta);
    if h ~= y(i)
    loss=loss+D(i,1);
    end
end
end

function [p_final,theta_final,loss_temp]=find_min(D,x,y,p,theta)
p_final=p(1,1);
theta_final=theta(1,1);
loss_temp=error(D,x,y,p_final,theta_final);
for i=1:1:2
    for j=1:1:10
        loss=error(D,x,y,p(i,1),theta(j,1));
        if loss<loss_temp
            loss_temp=loss;
            p_final=p(i,1);
            theta_final=theta(j,1);
        end
    end
end
end







