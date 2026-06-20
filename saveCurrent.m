function saveCurrent(fig, name, outDir)
    saveas(fig, fullfile(outDir, [name '.svg']));
    saveas(fig, fullfile(outDir, [name '.png']));
    savefig(fig, fullfile(outDir, [name '.fig']));
end
