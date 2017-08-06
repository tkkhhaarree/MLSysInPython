import os, requests
from bs4 import BeautifulSoup

# location where you have your movies stored
os.chdir("D:\\movi")
namelist=os.listdir()
for i in range (0, len(namelist)):
    namelist[i]=namelist[i].lower()
print(list)
print(namelist)

dirt=["480p", "720p", "1080p", "yify", "bluray", "dvd", "dvdrip", "dual audio", "hindi", "cd", "webrip", "end", "x264", "xvid", "hdrip", "brrip", "rip", "etrg",
      "mp4", "3gp", ".avi", "mkv", "filmywap", "src"]
for j in range (0, len(namelist)):
    for k in range (0, len(dirt)):
        if dirt[k] in namelist[j]:
            namelist[j]=namelist[j].replace(dirt[k], "")
            namelist[j]=namelist[j].replace(".", " ")
            namelist[j]=namelist[j].replace("_", " ")
            namelist[j] = namelist[j].replace("-", " ")
            namelist[j]=namelist[j].strip(" ")

for i in range (0, len(namelist)):
    searchurl="https://www.isubtitles.net/search?kwd="+namelist[i]
    res=requests.get(searchurl)
    if res is None:
        print("movie name "+namelist[i]+" not found!")
    else:
        soup=BeautifulSoup(res.text, "html.parser")
        c = soup.find("a")
        for j in range(0, 99):
            c = c.findNext("a")
            if c is None:
                break
            if c.parent.name=="h3":
                break
        if c is not None:
            movieurl = "https://www.isubtitles.net" + str(c["href"])
            movieurl=movieurl.replace("-subtitles", "/english-subtitles")
            res2=requests.get(movieurl)
            if res2 is None:
                 print("english subtitles for the movie "+namelist[i]+" not found!")
            else:
                soup2=BeautifulSoup(res2.text, "html.parser")
                c2=soup2.find("a")
                for k in range (0, 99):
                    c2=c2.findNext("a")
                    if c2 is None:
                        break
                    if c2.parent.name=="td":
                        break
                if c2 is not None:
                    suburl="https://www.isubtitles.net"+str(c2["href"])
                    res3=requests.get(suburl)
                    file=open(c.getText()+".rar", "wb")
                    file.write(res3.content)
                    print("subtitles for the movie: "+c.getText()+" downloaded successfully!")
                else:
                    print("english subtitles for the movie "+namelist[i]+" not found!")
        else:
            print("english subtitles for the movie "+namelist[i]+" not found!")
