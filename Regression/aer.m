function err = aer(Y,YHat)

err = abs(1 - YHat./Y);

end