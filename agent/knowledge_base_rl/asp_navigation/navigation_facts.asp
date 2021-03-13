location(s0).
room    (s1).
room    (s2).
room    (s3).
room    (s4).
location(s5).
room    (s6).
room    (s7).
location(s8).
room    (s9).
room    (s10).
location(s11).
room    (s12).
room    (s13).
location(s14).
room    (s15).
room    (s16).
location(s17).
location(s18).

% naming d1 for door on level 1. location where this door can be found after underscore
door(d0).
door(d1).
door(d2).
door(d3).
door(d4).
door(d5).

hasdoor(s1, d0).
hasdoor(s2, d1).
hasdoor(s3, d0).
hasdoor(s4, d1).
hasdoor(s9, d2).
hasdoor(s10, d2).
hasdoor(s6, d3).
hasdoor(s7, d3).
hasdoor(s12, d4).
hasdoor(s13, d4).
hasdoor(s15, d5).
hasdoor(s16, d5).

% Accecibility

acc(s0, s1). acc(s0, s2).acc(s1, s2).
acc(s5, s3). acc(s5, s6).
acc(s5, s8).
acc(s8, s4). acc(s8, s9).
acc(s8, s11).
acc(s11, s12).
acc(s11, s14).
acc(s14, s15).
acc(s17, s7). acc(s17, s10).
acc(s17, s18).
acc(s18, s13). acc(s18, s16).


dooracc(R1,D,R2) :- hasdoor(R1,D), hasdoor(R2,D), R1 != R2, door(D), room(R1), room(R2).
dooracc(R1,D,R2) :- dooracc(R2,D,R1).

acc(R,L) :- acc(L,R).
path(X, Y, I+1) :- at(X, I), at(Y, I+1), I=0..n-2.