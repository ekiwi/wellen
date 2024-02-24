library ieee;
use ieee.std_logic_1164.all;

entity test2 is 
end entity;

architecture foobar of test2 is
    type e is(foo, bar, tada);
    type e_array is array (1 downto 0) of e;
    signal bb: bit := '1';
    signal bv: bit_vector (7 downto 0) := "10101010";
    signal ii: integer := -12345;
    signal ff: real := 1.234;
    signal ee: e := foo;
    signal ea: e_array := (foo, tada);
    signal bbb: boolean := true;
begin
process 
begin
    wait for 50 ns;
    ee <= bar;
    ea(0) <= foo;
    bb <= '0';
    bv(5 downto 2) <= "1111";
    ii <= ii + 12345 + 87654;
    ff <= ff * (-2.0);
    bbb <= false;
    

    
    wait for 50 ns;
    ee <= tada;
    ea(1) <= ee;
    bb <= '0';
    bv <= "00000000";
    ii <= ii - 1;
    ff <= ff + 45.654;
    bbb <= not bbb;

    wait for 50 ns;
    assert false;
end process;
    

end architecture;

