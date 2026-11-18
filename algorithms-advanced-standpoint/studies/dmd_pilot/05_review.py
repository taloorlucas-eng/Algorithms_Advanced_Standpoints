"""Create a local blind photo-rating page. Open outputs/05_review.html with a browser."""
import csv,json,hashlib
from pathlib import Path
out=Path(__file__).resolve().parent/'outputs'
rows=list(csv.DictReader((out/'04_blind_key.csv').open()))
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真人实验盲评</title>
<style>body{font:17px system-ui;max-width:640px;margin:28px auto;padding:0 18px;color:#162537}img{display:block;width:220px;height:220px;margin:14px 0}fieldset{border:1px solid #ddd;border-radius:8px;margin:8px 0;padding:9px}label{display:inline-block;padding:7px}button{font:inherit;padding:8px 15px;margin:6px 6px 2px 0}#status{color:#265276}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ddd}</style>
<h1>微笑与眼镜盲评</h1><p>逐张看生成图。只有看见镜框、镜片或清晰的太阳镜才勾选“戴”。看不清请选“不确定”。完成后才显示方法名称。</p>
<p id="status"></p><img id="picture" alt="待判断的人脸图">
<fieldset id="sm"><legend>这张脸在微笑吗？</legend></fieldset><fieldset id="gl"><legend>戴眼镜了吗？</legend></fieldset>
<button id="back">上一张</button><button id="next">保存并继续</button><button id="download">导出已填 CSV</button><div id="score"></div>
<script>
const items=ITEMS, key="face_review_"+RUN; let answers={};try{answers=JSON.parse(localStorage.getItem(key)||"{}")}catch(e){}
let i=items.findIndex(r=>!answers[r.file]); if(i<0)i=0;
for(const [id,opts] of [["sm",["微笑","没有微笑","不确定"]],["gl",["戴","没戴","不确定"]]]){
const field=document.getElementById(id);opts.forEach((txt,j)=>{const l=document.createElement("label"),n=document.createElement("input");n.type="radio";n.name=id;n.value=["1","0","?"][j];l.append(n,txt);field.append(l)})}
function show(){const f=items[i].file;picture.src="blind/"+f;statusText().textContent="第 "+(i+1)+" / "+items.length+" 张，已完成 "+Object.keys(answers).length+" 张。";for(const n of document.querySelectorAll("input"))n.checked=answers[f]?.[n.name]===n.value;
if(Object.keys(answers).length===items.length)finish()}
function statusText(){return document.getElementById("status")}
function finish(){let groups={};for(const r of items){let k=r.w+" / "+r.arm, g=groups[k]??=[0,0,0,0];let a=answers[r.file];g[0]++;if(a.sm!=="?")g[1]++;if(a.sm==="1"&&a.gl!=="?"){g[2]++;if(a.gl==="1")g[3]++}}
document.getElementById("score").textContent="已完成。下面按 guidance / 方法显示：总数、可判表情数、微笑且可判眼镜数、其中戴眼镜数。";
let t=document.createElement("table");for(const [k,v] of Object.entries(groups)){let row=t.insertRow();[k,...v].forEach(x=>row.insertCell().textContent=x)}document.getElementById("score").append(t)}
document.getElementById("back").onclick=()=>{i=Math.max(0,i-1);show()};
document.getElementById("next").onclick=()=>{let a=document.querySelector('input[name="sm"]:checked'),b=document.querySelector('input[name="gl"]:checked');if(!a||!b){alert("请分别判断表情和眼镜。");return}answers[items[i].file]={sm:a.value,gl:b.value};try{localStorage.setItem(key,JSON.stringify(answers))}catch(e){}i=Math.min(i+1,items.length-1);show()};
document.getElementById("download").onclick=()=>{let lines=["file,smile,eyeglasses"];for(const r of items){let a=answers[r.file];if(a)lines.push([r.file,a.sm,a.gl].join(","))}let link=document.createElement("a");link.href=URL.createObjectURL(new Blob([lines.join("\\n")]));link.download="my_blind_labels.csv";link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)};show();
</script></html>'''
fingerprint=hashlib.sha256((out/'03_locked_choices.json').read_bytes()).hexdigest()[:12]
(out/'05_review.html').write_text(page.replace('ITEMS',json.dumps(rows)).replace('RUN',json.dumps(fingerprint)),encoding='utf8')
print('Open outputs/05_review.html. Do not open 04_blind_key.csv before rating.')
