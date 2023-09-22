import numpy
class Polynomial:
    def __init__(self,order,coefficients):
        self.order = order
        self.coefficients = coefficients
        self._reduce()

    def _reduce(self):
        if self.order == 0 & self.coefficients[0] == 0:
            return
        while self.coefficients[-1] == 0:
            self.order -= 1
            self.coefficients = self.coefficients[0:self.order+1]
            if self.order == 0:
                break

    def __repr__(self):
        string = ""
        order = self.order
        coefficients = self.coefficients
        # orders from 0 to order
        if (order == 0):
            return str(coefficients[0])
        if (coefficients[0] != 0):
            string = str(coefficients[0])
        for i in range(1,order+1):
            if coefficients[i] == 0:
                continue
            if coefficients[i] > 0:
                string += "+" + str(coefficients[i])+"*x^"+str(i)
            if coefficients[i] < 0:
                string += str(coefficients[i])+"*x^"+str(i)
        # return 0 for empty string
        if (string == ""):
            return str(0)
        return string
    
    def __eq__(self,other):
        order_self = self.order
        order_other = other.order
        # if order doesn't match
        if order_self != order_other:
            return False
        coefficients_self = self.coefficients
        coefficients_other = other.coefficients
        # if any coefficient doesn't match
        for i in range(order_self+1):
            if coefficients_self[i] != coefficients_other[i]:
                return False
        return True
    def __add__(self,other):
        order1 = self.order
        order2 = other.order
        coeff1 = self.coefficients
        coeff2 = other.coefficients
        # equal order
        if order1 == order2:
            coeff = numpy.add(coeff1,coeff2)
        # expand other
        if order1 > order2:
            coeff_new = [0]*(order1+1)
            for i in range(order2+1):
                coeff_new[i] = coeff2[i]
            coeff = numpy.add(coeff1,coeff_new)
        # expand self
        if order1 < order2:
            coeff_new = [0]*(order2+1)
            for i in range(order1+1):
                coeff_new[i] = coeff1[i]
            coeff = numpy.add(coeff2,coeff_new)
        order = len(coeff)-1
        return Polynomial(order,coeff)

    def __neg__(self):
        self.coefficients = numpy.multiply(self.coefficients,-1)
        return Polynomial(self.order,self.coefficients)
    
    def __sub__(self, other):
        return self + (-other)

    def __mul__(self,other):
        order_self = self.order
        order_other = other.order
        coeff_self = self.coefficients
        coeff_other = other.coefficients
        order_new = order_self+order_other
        coeff_new = [0]*(order_new+1)
        for i in range(order_self+1):
            for j in range(order_other+1):
                coeff_new[i+j] += coeff_self[i]*coeff_other[j]
        return Polynomial(order_new,coeff_new)
    @staticmethod
    def from_string(string):
        poly_string = string.split()
        # highest order
        order = 0
        coefficients = [0]
        sign = 1
        for monomial in poly_string:
            # sign
            if (monomial == "-"):
                sign = -1
                continue
            if (monomial == "+"):
                sign = 1
                continue
            # constant
            if monomial.isnumeric():
                coefficients[0] += sign*int(monomial)
                continue
            # negative constant
            if monomial[1:].isnumeric():
                coefficients[0] += sign*int(monomial)
                continue
            # regular terms a*x^b
            monomial = monomial.split("^")
            # if no "^"" found
            if len(monomial) == 1:
                term_order = 1
            else:
                term_order = int(monomial[1])
            monomial = monomial[0]
            # expand coefficients array
            if (term_order > order):
                order_old = order
                order = term_order
                temp = coefficients
                coefficients = [0]*(order+1)
                for i in range(order_old+1):
                    coefficients[i] = temp[i]
            monomial = monomial.split("*")
            # x or -x
            if len(monomial) == 1:
                if monomial[0] == "x":
                    coefficient = 1
                if monomial[0] == "-x":
                    coefficient = -1    
            else:
                coefficient = monomial[0]
            coefficients[term_order] += sign*int(coefficient)
        return Polynomial(order,coefficients)
    
