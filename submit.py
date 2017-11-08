import os
import sys

if __name__=="__main__":
	f = raw_input("File:")
	m = raw_input("Commit message:")
	os.system("git add "+f)
	os.system("git commit -m \""+m+"\"")
	os.system("git push origin master")