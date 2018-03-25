const request = require('request');
const cheerio = require('cheerio');
const URL = require('url-parse');
const fs = require('fs')
const csvWriter = require('csv-write-stream')

const parseBody = (body) => {
    const $ = cheerio.load(body)

    const bokes = []
    $('.boke').each(function(i, element) {
        const text = $(this).find('.boke-text').text().trim()
        if (!text) return
        const img = "https:" + $(this).find('.boke-photo img').attr('src')
        const filename = img.split('/').slice(-1).pop()
        bokes.push({ text, img, filename })
    })

    return bokes
}

const saveImage = (boke) => {
    const filename = "data/images/" + boke.filename

    return new Promise((resolve, reject) => {
        request.head(boke.img, function(err, res, body) {
            console.log(boke.img)
            console.log('content-type:', res.headers['content-type']);
            console.log('content-length:', res.headers['content-length']);
            request(boke.img).pipe(fs.createWriteStream(filename)).on('close', resolve);
        })
    })
}

const crawl = (baseUrl, last, page=1) => {
    if (page > last) return
    const url = baseUrl + page
    console.log("crawl: " + url)
    request(url, (error, response, body) => {
        const bokes = parseBody(body)
        bokes.forEach(async (boke) => await saveImage(boke))
        const writer = csvWriter()
        writer.pipe(fs.createWriteStream(`data/p${page}.csv`))
        bokes.forEach((b) => writer.write(b))
        writer.end()
        setTimeout(() => crawl(baseUrl, last, page+1), 1000)
    })
}

// const baseUrl = "https://bokete.jp/boke/popular?page="
// const baseUrl = "https://bokete.jp/boke/hot?page="
// const baseUrl = "https://bokete.jp/boke/select?page="
//  const baseUrl = "https://bokete.jp/boke/pickup?page="
const baseUrl = "https://bokete.jp/boke/legend?page="
crawl(baseUrl, 8)