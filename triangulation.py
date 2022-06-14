import pickFileName as pickfilename
import returnBothCamImgPoints as getimgpoints
import readXML as readxmldata
import cv2 as cv
import numpy as np
import pandas as pd

def triangulationPoints_test(leftcmtx, rightcmtx, leftdvec, rightdvec, R, T, size,
                                    leftimgpoints, rightimgpoints, lrvec, ltvec, rrvec, rtvec):

    R1, R2, P1, P2, Q ,test1,test2= cv.stereoRectify(leftcmtx, leftdvec, rightcmtx, rightdvec, size, R, T)
    ul_imgpoints = cv.undistortPoints(leftimgpoints,leftcmtx, leftdvec, None, R1,P1)
    ur_imgpoints = cv.undistortPoints(rightimgpoints, rightcmtx, rightdvec, None, R2, P2)
    xl4 = cv.triangulatePoints(P1,P2,ul_imgpoints,ur_imgpoints)
    xl3 = np.zeros(shape = (3,1), dtype = np.float32)

    for i in range(len(xl3)):
        xl3[i][0] = xl4[i][0] / xl4[3][0]

    xl3 = np.dot(np.linalg.inv(R1),xl3)
    xr3 = np.dot(R,xl3) + T
    fakervec = np.zeros(shape = (1,3) , dtype = np.float32)
    faketvec = np.zeros(shape = (1,3) , dtype = np.float32)

    leftProPoints,test2 = cv.projectPoints(xl3, fakervec, faketvec, leftcmtx, leftdvec, leftimgpoints)
    rightProPoints,test4 = cv.projectPoints(xr3, fakervec, faketvec, rightcmtx, rightdvec, rightimgpoints)

    triangulationError = np.zeros(shape = (2,2), dtype = np.float32)

    triangulationError[0][0] = leftProPoints[0][0][0] - leftimgpoints[0][0][0]
    triangulationError[0][1] = leftProPoints[0][0][1] - leftimgpoints[0][0][1]
    triangulationError[1][0] = rightProPoints[0][0][0] - rightimgpoints[0][0][0]
    triangulationError[1][1] = rightProPoints[0][0][1] - rightimgpoints[0][0][1]


    worldCoor = toWorldCoordinate(xl3, xr3, lrvec, ltvec, rrvec, rtvec)
    #print(triangulationError)
    #print(worldCoor)
    return worldCoor, triangulationError



def toWorldCoordinate(xl3,xr3,lrvec, ltvec, rrvec, rtvec):

    fix = np.array([0,0,0,1],dtype = np.float32)
    fix2 = np.array([1], dtype = np.float32)

    lrvecRod = cv.Rodrigues(lrvec)
    rtl = np.hstack((lrvecRod[0],ltvec))
    RT_L = np.vstack((rtl,fix))

    #print(RT_L)

    rrvecRod = cv.Rodrigues(rrvec)
    rtr = np.hstack((rrvecRod[0],rtvec))
    RT_R = np.vstack((rtr, fix))
    #print(RT_R)

    xl3 = np.vstack((xl3,fix2))
    xr3 = np.vstack((xr3,fix2))
    wl3 = np.dot(np.linalg.inv(RT_L), xl3)
    wr3 = np.dot(np.linalg.inv(RT_R), xr3)
    #print(xl3, xr3)


    return wl3[0:3]



if __name__ == "__main__":
    #get calibrate data
    leftxmlDataPath = "F:/ansonExperiment_2/calibratePhoto/calibrateWindTurbine0329/leftExtrinsic_intrinsicResult/leftResult.xml"
    rightxmlDataPath = "F:/ansonExperiment_2/calibratePhoto/calibrateWindTurbine0329/rightExtrinsic_intrinsicResult/rightResult.xml"

    leftcmtx, leftdvec, leftrvec, lefttvec = readxmldata.getCalibrationData(leftxmlDataPath)
    rightcmtx, rightdvec, rightrvec, righttvec = readxmldata.getCalibrationData(rightxmlDataPath)
    h,w = leftrvec.shape
    leftrvec = np.reshape(leftrvec, (w,h))
    h,w = lefttvec.shape
    lefttvec = np.reshape(lefttvec, (w,h))
    h,w = leftdvec.shape
    leftdvec = np.reshape(leftdvec, (w,h))

    h,w = rightrvec.shape
    rightrvec = np.reshape(rightrvec, (w,h))
    h,w = righttvec.shape
    righttvec = np.reshape(righttvec, (w,h))
    h,w = rightdvec.shape
    rightdvec = np.reshape(rightdvec, (w,h))



    Rvec, Tvec = readxmldata.R_Tcalculator2(leftrvec,lefttvec, rightrvec, righttvec)
    #print(Rvec,Tvec)
    #get image points of both camera.
    markersize = 6
    totalmarker = 250
    amount = 3

    size = (int(2048), int(2046))
    #excelfilename = "F:\ThesisDocuments\program\pickFileName\detectedResult.xlsx"
    excelfilename = "big_static.xlsx"
    fileCantUse = pickfilename.findMatchFileName(amount, excelfilename)

    leftimgPoints , rightimgPoints= getimgpoints.detectArucoGetImgPoints(excelfilename, amount,markersize, totalmarker )
    print(leftimgPoints)
    #print(rightimgPoints)
    for i in range(len(leftimgPoints)):
        leftimgpoints_id = leftimgPoints[i]
        rightimgpoints_id = rightimgPoints[i]
        save3dpoints = []
        saveTrianError = []
        for ii in range(len(leftimgpoints_id)):
            saveleftimgpoints = np.zeros(shape = (1,1,2), dtype = np.float32)
            saverightimgpoints = np.zeros(shape = (1,1,2), dtype = np.float32)
            saveleftimgpoints[0][0][0] = leftimgpoints_id[ii][0]
            saveleftimgpoints[0][0][1] = leftimgpoints_id[ii][1]
            saverightimgpoints[0][0][0] = rightimgpoints_id[ii][0]
            saverightimgpoints[0][0][1] = rightimgpoints_id[ii][1]
            worldCoor, trianError = triangulationPoints_test(leftcmtx, rightcmtx,leftdvec, rightdvec, Rvec, Tvec,
                                                        size, saveleftimgpoints, saverightimgpoints,
                                                        leftrvec, lefttvec, rightrvec, righttvec)
            worldCoor = np.reshape(worldCoor, (1,3))

            save3dpoints.append(worldCoor)
            saveTrianError.append(trianError)

        input = {}
        lcx = []
        lcy = []
        lcz = []
        leftErrx = []
        leftErry = []
        rightErrx = []
        rightErry = []
        for j in range(len(save3dpoints)):
            lcx.append(save3dpoints[j][0][0])
            lcy.append(save3dpoints[j][0][1])
            lcz.append(save3dpoints[j][0][2])

        for j in range(len(saveTrianError)):
            leftErrx.append(saveTrianError[j][0][0])
            leftErry.append(saveTrianError[j][0][1])
            rightErrx.append(saveTrianError[j][1][0])
            rightErry.append(saveTrianError[j][1][1])


        input["lcx"] = lcx
        input["lcy"] = lcy
        input["lcz"] = lcz
        input["leftErrx"] = leftErrx
        input["leftErry"] = leftErry
        input["rightErrx"] = rightErrx
        input["rightErry"] = rightErry

        tosave = pd.DataFrame(input)
        filename = str(i)+"_3dpoints.xlsx"
        writer = pd.ExcelWriter(filename, engine = "openpyxl")
        tosave.to_excel(writer)
        writer.save()


