library ieee;
use ieee.std_logic_1164.all;

entity test is 
end entity;

architecture foobar of test is
    type e is(foo, bar);
    type r is record
        a: std_logic;
        b: std_logic_vector(5 downto 2);
        c: std_logic_vector(1 to 4);
        d: e;
    end record;
    signal ee: e := foo;
    signal rr: r;
begin
process 
begin
    wait for 50 ns;
    ee <= bar;
    rr.a <= 'U';
    rr.b <= "HLZ-";
    rr.c <= "WX10";
    rr.d <= foo;
    
    wait for 50 ns;
    ee <= foo;
    rr.a <= '1';
    rr.b <= "1010";
    rr.c <= "0101";
    rr.d <= bar;

    wait for 50 ns;
    assert false;
end process;
    

end architecture;

