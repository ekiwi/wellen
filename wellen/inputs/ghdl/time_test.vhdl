library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity Time_Test is
end entity Time_Test;

architecture Behavioral of Time_Test is
    signal t1 : time := 500 ns;
    signal t2 : time := 2 sec;
    signal t3 : time;
begin
    process
    begin
        t3 <= t1 + t2;
        report "t1 = " & time'image(t1);
        report "t2 = " & time'image(t2);
        report "t3 = " & time'image(t3);
        wait for 10 ns; -- Allow time to pass for signal updates
    end process;
end architecture Behavioral;

