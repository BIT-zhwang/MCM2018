import xlwt
workbook=xlwt.Workbook()
worksheet = workbook.add_sheet('My Worksheet')
worksheet.write(0, 0, 0.55)
worksheet.write(0, 1, 0.45)
workbook.save('Excel_Workbook.xls')