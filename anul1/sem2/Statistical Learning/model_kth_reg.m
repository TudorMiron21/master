function [param,  Er]=model_kth_reg(x,y,lambda,k)

if length(x)~=length(y)
    error('Date incorecte; lungimi diferite ale vectorilor');
 
else
    A=zeros(k+1);
    B=zeros(k+1,1);
    for i=1:k+1
        for j=i:k+1
            A(i,j)=sum(x.^(2*k+2-i-j));
            A(j,i)=A(i,j);
        end
       B(i,1)=sum((x.^(k+1-i)).*y); 
    end
    for i=1:k+1
        A(i,i)=A(i,i)+lambda;
    end
    param=linsolve(A,B);
    %param=pinv(A)*B;
    u=linspace(min(x),max(x),300);
    vv=zeros(k+1,300);
    for i=1:k+1
        vv(i,:)=u.^(k+1-i);
    end

    v=param'*vv;
    figure
    plot(x,y,'o','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',7);
    hold on
    plot(u,v,'b','Linewidth',3);
    title(sprintf('Modelul %d-th de regresie',k));
    legend('Data',sprintf('Modelul %d-th',k))
   
end
xx=zeros(k+1,length(x));
    for i=1:k+1
        xx(i,:)=x.^(k+1-i);
    end

Er=sqrt(sum((param'*xx-y').^2)/length(x));% root mean square error

end
    