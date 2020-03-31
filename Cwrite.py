from xlrd import open_workbook
rb = open_workbook('C:\Users\Leibniz\Desktop\predict1.xlsx')
#通过get_sheet()获取的sheet有write()方法
ws = rb.get_sheet(0)
ws.write(0, 0, 'changed!')
 
rb.save('C:\Users\Leibniz\Desktop\predict1.xlsx')
rb.close()
