library ieee;
use ieee.std_logic_1164.all;

entity ali is 
end entity;

architecture foobar of ali is
    signal large : std_logic_vector (6 downto 0) := "0000000";
    signal full : std_logic_vector (6 downto 0);
    signal msb4 : std_logic_vector (3 downto 0);
    signal lsb3 : std_logic_vector (2 downto 0);
    signal middle_a_4 : std_logic_vector (3 downto 0);
    signal middle_b_3 : std_logic_vector (2 downto 0);
    signal bit0 : std_logic;
    signal bit6 : std_logic;
    signal bit4 : std_logic;
    signal mix  : std_logic_vector (3 downto 0);
begin
  msb4 <= large(6 downto 3);
  lsb3 <= large(2 downto 0);
  middle_a_4 <= large(5 downto 2);
  middle_b_3 <= large(3 downto 1);
  bit0 <= large(0);
  bit6 <= large(6);
  bit4 <= large(4);
  mix(3 downto 2) <= large(5 downto 4);
  mix(1 downto 0) <= large(2 downto 1);
  full <= large;
process 
begin
    wait for 50 ns;
    large <= "1011001";    
    wait for 50 ns;
    large <= "1000101";
    wait for 50 ns;
    large <= "1010101";
    wait for 50 ns;
    assert false;
end process;
    

end architecture;

