function writeCorrCSV(R, filename, symbols)
    n = length(symbols);
    cellOut = cell(n+1, n+1);
    cellOut{1,1} = '';
    for i = 1:n
        cellOut{1, i+1} = symbols{i};
        cellOut{i+1, 1} = symbols{i};
    end
    for i = 1:n
        for j = 1:n
            cellOut{i+1, j+1} = sprintf('%.4f', R(i,j));
        end
    end
    fid = fopen(filename, 'w');
    for r = 1:n+1
        fprintf(fid, '%s', cellOut{r,1});
        for c = 2:n+1
            fprintf(fid, ',%s', cellOut{r,c});
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end
