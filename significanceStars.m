function s = significanceStars(p)
    if p < 0.001
        s = '***';
    elseif p < 0.01
        s = '** ';
    elseif p < 0.05
        s = '*  ';
    else
        s = 'n.s.';
    end
end
