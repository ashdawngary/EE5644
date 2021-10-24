% decide for 1a/b log likelihood test.
function desc = decide(x, gamma)
    desc = loglikeratio(x) >= log(gamma);
end