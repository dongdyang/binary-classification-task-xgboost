from sklearn.feature_extraction.text import TfidfVectorizer

class Utils:

    def is_float(self, s):
        temp = sum([n.isdigit() for n in s.strip().split('.')])
        return temp == 2

    def processTime(self, time):
        time = str(time)
        if time.find("2018")!=-1:
            return 1
        elif time.find("2017")!=-1:
            return 0
        else:
            return None

    def processBook(self, book):
        if 0<book<=1500:
            return int(book/100)
        elif book>1500:
            return 15
        else:
            return None

    def processfNumber(self, number):
        number = str(number)
        if self.is_float(number) or number.isdigit():
            temp = int(float(number))
            if temp>1500:
                return 1500
            elif temp<0:
                return 0
            else:
                return int(temp)
        else:
            return None

    def processRain(self, rain):
        rain = str(rain)
        if self.is_float(rain) or rain.isdigit():
            return int(float(rain)/10)
        elif rain.find("inch")!=-1:
            return int(float(rain.split()[0])/10)
        elif rain.find("mm")!=-1:
            digit = ""
            for ch in rain:
                if ch.isdigit():
                    digit+=ch
                else:
                    break
            return int(float(digit) * 0.0393701 / 10)
        else:
            return None


    def processTemp(self, temp):
        temp = str(temp)
        def findFCK(t):
            if not t:
                return 0
            for ch in t:
                if ch == "K" or ch == "k":
                    return 3
                if ch == "C" or ch == "c":
                    return 2
                if ch == "F" or ch == "f":
                    return 1
            return 0

        if not temp:
            return None
        if self.is_float(temp) or temp.isdigit():
            return int(float(temp)/10)
        else:
            flag = findFCK(temp)
            if flag:
                digit = ""
                for ch in temp:
                    if ch.isdigit():
                        digit += ch
                    else:
                        break
                if digit and flag == 1:
                    digit = str(digit)
                elif digit and flag == 2:
                    digit = str(float(digit) * 1.8 + 32)
                elif digit and flag == 3:
                    digit = str(float(digit) * 1.8 - 459.67)
                return int(float(digit)/10) if digit != "" else None
            else:
                return None