#!/usr/bin/env python

# This program calculates letter grades

x = input("Enter a decimal between 0 and 1: ")
y = float(x)

if y >= 0.9:
    print('A')
elif y >= 0.8:
    print('B')
elif y >= 0.7:
    print('C')
elif y >= 0.6:
    print('D')
elif y < 0.6:
    print('F')
