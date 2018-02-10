from __future__ import division

import cv2
import numpy as np
from neuralNetwork import *
from scipy.spatial import distance




# def is_same_objectGreen(region):
#     if len(greenLineRegions)==0:
#         greenLineRegions.append(region)
#         return False
#     for r in range(0, len(greenLineRegions)):
#         if are_same_regions(region, greenLineRegions[r]):
#             greenLineRegions[r] = region
#             return True
#     greenLineRegions.append(region)
#     return False
#
#
def is_colliding_with_greenLine():
    maxX = max((greenLine[0], greenLine[2]))
    minX = min((greenLine[0], greenLine[2]))
    maxY = max((greenLine[1], greenLine[3]))
    minY = min((greenLine[1], greenLine[3]))
    slope = (greenLine[3]-greenLine[1])/(maxX-minX)
    b = greenLine[1] - slope*greenLine[0]


    for rg in range(0, len(allNumbers)):
        if allNumbers[rg][2] == 0:
            x = allNumbers[rg][1][0]
            y = allNumbers[rg][1][1]
            w = allNumbers[rg][1][2]
            h = allNumbers[rg][1][3]
            for x1 in range(x, x+w+1):
                if x1>=minX and x1<=maxX:
                    if y==int(round(slope*x1+b)) or y+h == int(round(slope*x1+b)):
                        allNumbers[rg][2] = 1
                        return True
            if allNumbers[rg][2]==0:
                for y1 in range(y, y+h+1):
                    if y1 >= minY and y1 <= maxY:
                        if y1 == int(round(slope*x+b)) or y1 == int(round(slope*(x+w)+b)):
                            allNumbers[rg][2] = 1
                            return True
    return False

def is_colliding_with_blueLine():
    maxX = max((blueLine[0], blueLine[2]))
    minX = min((blueLine[0], blueLine[2]))
    maxY = max((blueLine[1], blueLine[3]))
    minY = min((blueLine[1], blueLine[3]))
    slope = (blueLine[3] - blueLine[1]) / (maxX - minX)
    b = blueLine[1] - slope * blueLine[0]



    for rg in range(0, len(allNumbers)):
        if allNumbers[rg][3] == 0:
            x = allNumbers[rg][1][0]
            y = allNumbers[rg][1][1]
            w = allNumbers[rg][1][2]
            h = allNumbers[rg][1][3]
            for x1 in range(x+5, x + w + -4):
                if x1 >= minX and x1 <= maxX:
                    if y == int(round(slope * x1 + b)) or (y + h) == int(round(slope * x1 + b)):
                        allNumbers[rg][3] = 1
                        return True
            if allNumbers[rg][3] == 0:
                for y1 in range(y+5, y + h + 4):
                    if y1 >= minY and y1 <= maxY:
                        if y1 == int(round(slope * x + b)) or y1 == int(round(slope * (x + w) + b)):
                            allNumbers[rg][3] = 1
                            return True

    return False

def adjustLines():
    maxX = max((blueLine[0], blueLine[2]))
    minX = min((blueLine[0], blueLine[2]))
    maxY = max((blueLine[1], blueLine[3]))
    minY = min((blueLine[1], blueLine[3]))
    slope = (blueLine[3] - blueLine[1]) / (maxX - minX)
    b = blueLine[1] - slope * blueLine[0]

    blueLine[0] = minX + 5
    blueLine[1] = int(round(slope * blueLine[0] + b))
    blueLine[2] = maxX - 5
    blueLine[3] = int(round(slope * blueLine[2] + b))

    maxX = max((greenLine[0], greenLine[2]))
    minX = min((greenLine[0], greenLine[2]))
    maxY = max((greenLine[1], greenLine[3]))
    minY = min((greenLine[1], greenLine[3]))
    slope = (greenLine[3]-greenLine[1])/(maxX-minX)
    b = greenLine[1] - slope*greenLine[0]

    greenLine[0] = minX + 5
    greenLine[1] = int(round(slope * greenLine[0] + b))
    greenLine[2] = maxX - 5
    greenLine[3] = int(round(slope * greenLine[2] + b))

    #
    # if y < minY or y > maxY:
    #     return False
    # if x < minX or x > maxX:
    #     return False
    #
    # if (greenLine[2]-greenLine[0])==0:
    #     if x==greenLine[0]:
    #         if y>=minY and y<=maxY:
    #             print('PIP ', x, ' ', y)
    #             return True
    #     return False
    # slope = (greenLine[3]-greenLine[1])/(maxX-minX)
    # b = greenLine[1] - slope*greenLine[0]
    # if y == int(round(slope*x+b)):
    #     return True
    # return False

def calculateSum():
    sum = 0

    for s in range(0, len(allNumbers)):
        if allNumbers[s][3]==1:
            sum += allNumbers[s][0]

        if allNumbers[s][2]==1:
            sum -= allNumbers[s][0]

    return sum

def isSameRegion(old, new):

    # if old[0] != getNumberFromRegion(new[0][0]):
    #     return False

    oldX = old[1][0]
    oldY = old[1][1]
    newX = new[0][1][0]
    newY = new[0][1][1]

    if newX>=oldX-5 and newX<=oldX+5:
        if newY>=oldY-5 and newY<=oldY+5:
            return True

    return False

def isSameNumber(num1, num2):
    if num1 == num2:
        return True
    #
    # if num1 == 9 or num1 == 4 or num1==1 or num1==3:
    #     if num2==9 or num2==4 or num2==1 or num2==3:
    #         return True
    # if num1==1 or num1==7:
    #     if num2==1 or num2==7:
    #         return True

    # if num1==8 or num1==9 or num1==0 or num1==6 or num1==2 or num1==3:
    #     if num2 == 8 or num2 == 9 or num2 == 0 or num2==6 or num2==2 or num2==3:
    #         return True

    # if num1==8 or num1==0 or num1==2 or num1==3:
    #     if num2 == 8 or num2==0 or num2==2 or num2==3:
    #         return True



    if num1==3 or num1==5:
        if num2 == 3 or num2 == 5:
            return True

    return False


def findFake(temp):
    okolina = []
    for jj in range(0, len(allNumbers)):
        if allNumbers[jj][4]>0 and allNumbers[jj][1][0]<650 and allNumbers[jj][1][1]<470 and isSameNumber(allNumbers[jj][0], temp[2]):
            x1 = temp[0][1][0]
            y1 = temp[0][1][1]
            x2 = allNumbers[jj][1][0]
            y2 = allNumbers[jj][1][1]
            dist = distance.euclidean([x1, y1], [x2, y2])
            okolina.append([allNumbers[jj], dist])


    minimalni = 100000
    fejk = None
    if len(okolina)>0:
        for jjj in range(0, len(okolina)):
            if okolina[jjj][1]<minimalni:
                minimalni = okolina[jjj][1]
                fejk = okolina[jjj][0]

    if fejk != None and minimalni<fejk[4]*2:
        #print fejk[0], temp[2]
        for ind in range(0, len(allNumbers)):
            x1 = allNumbers[ind][1][0]
            y1 = allNumbers[ind][1][1]
            x2 = fejk[1][0]
            y2 = fejk[1][1]
            if x1==x2 and y1==y2:
                allNumbers[ind][1][0] = temp[0][1][0]
                allNumbers[ind][1][1] = temp[0][1][1]
                allNumbers[ind][1][2] = temp[0][1][2]
                allNumbers[ind][1][3] = temp[0][1][3]
                allNumbers[ind][4] = 0
                return True

    okolina2 = []
    for jj2 in range(0, len(allNumbers)):
        if allNumbers[jj2][4] > 0 and allNumbers[jj2][1][0] < 650 and allNumbers[jj2][1][1] < 470:
            x1 = temp[0][1][0]
            y1 = temp[0][1][1]
            x2 = allNumbers[jj2][1][0]
            y2 = allNumbers[jj2][1][1]
            dist = distance.euclidean([x1, y1], [x2, y2])
            okolina2.append([allNumbers[jj2], dist])

    minimalni2 = 100000
    fejk2 = None
    if len(okolina2) > 0:
        for jjj2 in range(0, len(okolina2)):
            if okolina2[jjj2][1] < minimalni2:
                minimalni2 = okolina2[jjj2][1]
                fejk2 = okolina2[jjj2][0]

    if fejk2 != None and minimalni2<40:
        # print fejk[0], temp[2]
        for ind2 in range(0, len(allNumbers)):
            x1 = allNumbers[ind2][1][0]
            y1 = allNumbers[ind2][1][1]
            x2 = fejk2[1][0]
            y2 = fejk2[1][1]
            if x1 == x2 and y1 == y2:
                allNumbers[ind2][1][0] = temp[0][1][0]
                allNumbers[ind2][1][1] = temp[0][1][1]
                allNumbers[ind2][1][2] = temp[0][1][2]
                allNumbers[ind2][1][3] = temp[0][1][3]
                allNumbers[ind2][4] = 0
                return True

    return False

def updateRegions(newRegions):
    if len(allNumbers)==0:
        for kk in range(0, len(newRegions)):
            # broj u regionu, pozicija regiona, prosao kroz zelenu, prosao kroz plavu, fejk
            allNumbers.append([getNumberFromRegion(newRegions[kk][0]), newRegions[kk][1], 0, 0, 0])
            a = is_colliding_with_greenLine()
            b = is_colliding_with_blueLine()
            return b, a

    temp = []
    for ii in range(0, len(newRegions)):
        #drugi param govori da li je pronasao svog para iz prethodnog frejma, ako nije (0)
        #onda je to novi region koji se pojavio
        temp.append([newRegions[ii], 0, getNumberFromRegion(newRegions[ii][0])])



    for iii in range(0, len(allNumbers)):
        found = 0
        for j in range(0, len(temp)):
            if temp[j][1] == 0 and isSameNumber(allNumbers[iii][0], temp[j][2]) and isSameRegion(allNumbers[iii], temp[j]):
                allNumbers[iii][1] = temp[j][0][1]
                allNumbers[iii][4] = 0
                temp[j][1] = 1
                found = 1
                break

        found1 = 0
        if found == 0:
            for jw in range(0, len(temp)):
                if temp[jw][1] == 0 and isSameRegion(allNumbers[iii], temp[jw]):
                    allNumbers[iii][1] = temp[jw][0][1]
                    allNumbers[iii][4] = 0
                    temp[jw][1] = 1
                    found1 = 1
                    break

        if found == 0 and found1 == 0:
            allNumbers[iii][1][0] += 1
            allNumbers[iii][1][1] += 1
            allNumbers[iii][4] += 1


    for iiii in range(0, len(temp)):
        if temp[iiii][1] == 0:
            if temp[iiii][0][1][0]<10 or temp[iiii][0][1][1]<10:
                allNumbers.append([getNumberFromRegion(temp[iiii][0][0]), temp[iiii][0][1], 0, 0, 0])
            else:
                #findFake(temp[iiii])
                   #allNumbers.append([getNumberFromRegion(temp[iiii][0][0]), temp[iiii][0][1], 0, 0, 0])
                if not findFake(temp[iiii]):
                    allNumbers.append([getNumberFromRegion(temp[iiii][0][0]), temp[iiii][0][1], 0, 0, 0])

    #ukloni one koji su izasli iz frejma
    # skl = 0
    # while skl<len(allNumbers):
    #     if allNumbers[skl][1][0]>680 or allNumbers[skl][1][1]>490:
    #         allNumbers.remove(allNumbers[skl])
    #     else:
    #         skl +=1

    a = is_colliding_with_greenLine()
    b = is_colliding_with_blueLine()
    return b, a





file = open('videos/out.txt','w')
file.write('RA 70/2013 Stefan Radanovic\n')
file.write('file    sum')
for f in range(10):
    filename = 'videos/video-' + str(f) + '.avi'
    cap = cv2.VideoCapture(filename)
    i = 0
    allNumbers = []
    greenLine = [0, 0, 0, 0]
    blueLine = [0, 0, 0, 0]
    greenLineRegions = []


    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()


        if ret==False:
            break

        frame2 = frame
        frame3 = frame


        i+=1
        if i==1:

            probaa = frame
            #izdvoji samo zelenu liniju
            lower = np.array([0, 100, 0], dtype="uint8")
            upper = np.array([60, 255, 50], dtype="uint8")
            frame = cv2.inRange(frame, lower, upper)

            #hough transformacija, trazenje linija
            img = frame
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (5, 5))
            dil = cv2.dilate(edges, kernel, iterations=3)
            kernel = np.ones((5, 5))
            edges = cv2.erode(dil, kernel, iterations=2)
            minLineLength = 50
            maxLineGap = 0
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
            #print len(lines)
            greenLine = lines[0][0]
            #print lines[0][0]
            maxDuzina = 0

            for k in range(0, len(lines)):
                for x1, y1, x2, y2 in lines[k]:
                    duzina = x2-x1
                    if duzina>maxDuzina:
                        maxDuzina = duzina
                        greenLine = lines[k][0]

            # izdvoji samo plavu liniju
            lower = np.array([100, 0, 0], dtype="uint8")
            upper = np.array([255, 50, 50], dtype="uint8")
            frame2 = cv2.inRange(frame2, lower, upper)

            # hough transformacija, trazenje linija
            img = frame2
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (5, 5))
            dil = cv2.dilate(edges, kernel, iterations=2)
            kernel = np.ones((5, 5))
            edges = cv2.erode(dil, kernel, iterations=1)
            minLineLength = 50
            maxLineGap = 0
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
            #print len(lines)
            blueLine = lines[0][0]
            #print lines[0][0]
            maxDuzina = 0

            for k in range(0, len(lines)):
                for x1, y1, x2, y2 in lines[k]:
                    duzina = x2 - x1
                    if duzina > maxDuzina:
                        maxDuzina = duzina
                        blueLine = lines[k][0]

            adjustLines()
            # cv2.line(probaa, (blueLine[0], blueLine[1]), (blueLine[2], blueLine[3]), (0, 255, 255), 2)
            # cv2.line(probaa, (greenLine[0], greenLine[1]), (greenLine[2], greenLine[3]), (255, 0, 255), 2)
            #
            # cv2.imwrite('vid' + str(f) + '.jpg', probaa)

        if i>0:
        #if i==532 or i==485:
            #print(i)
            image_color = frame3
            img = invert(image_bin(image_gray(image_color)))
            img_bin = erode(dilate(img))
            selected_regions, regions, regions_original = select_roi_from_video(image_color.copy(), img)
            if len(regions_original)>0:
                a, b = updateRegions(regions_original)
                # if a == True or b == True:
                #     print '============================='
                #     print 'suma ' + str(calculateSum())
                #     for skj in range(0, len(allNumbers)):
                #         if allNumbers[skj][1][0]<650 and allNumbers[skj][1][1]<450:
                #             # print 'broj: ', allNumbers[skj][0]
                #             # print 'poz: x=', allNumbers[skj][1][0], ' y=', allNumbers[skj][1][1]
                #             # print 'fejk: ', allNumbers[skj][4]
                #             # print '----------------------------'
                #             cv2.rectangle(selected_regions, (allNumbers[skj][1][0], allNumbers[skj][1][1]), (allNumbers[skj][1][0] + allNumbers[skj][1][2], allNumbers[skj][1][1] + allNumbers[skj][1][3]), (0, 0, 255), 1)
                #     plt.imshow(selected_regions)
                #     plt.show()

            # if i%50 == 0:
            #     print '============================='
            #     print '============================='
            #     for skj in range(0, len(allNumbers)):
            #         if allNumbers[skj][1][0]<650 and allNumbers[skj][1][1]<450:
            #             print 'broj: ', allNumbers[skj][0]
            #             print 'poz: x=', allNumbers[skj][1][0], ' y=', allNumbers[skj][1][1]
            #             print 'fejk: ', allNumbers[skj][4]
            #             print '----------------------------'
            #             cv2.rectangle(selected_regions, (allNumbers[skj][1][0], allNumbers[skj][1][1]), (allNumbers[skj][1][0] + allNumbers[skj][1][2], allNumbers[skj][1][1] + allNumbers[skj][1][3]), (0, 0, 255), 1)
            #     plt.imshow(selected_regions)
            #     plt.show()




    cap.release()
    cv2.destroyAllWindows()
    string = 'video-' + str(f) + '.avi ' + str(calculateSum()) + '\n'
    print string
    # for jh in range(0, len(allNumbers)):
    #     print(allNumbers[jh][0])
    file.write(string)
file.close()