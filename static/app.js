
const overviewDiv = document.getElementById('overview_result');
const detailsDiv = document.getElementById('details_result');
const userCam = document.getElementById('user_cam');
const statusEl = document.getElementById('status');

function renderSummary(data){
  if(!data || !data.overview){ overviewDiv.innerHTML=''; detailsDiv.innerHTML=''; return; }
  const ov = data.overview;

  const makeTagGroup = (title, obj, icon)=>{
    if(!obj || !Object.keys(obj).length) return '';
    let tags = Object.entries(obj)
      .map(([k,v])=>`<div class="tag">${k}<span class="count">${v}</span></div>`)
      .join('');
    return `<div class="group-block"><div class="group-heading">${icon||''}${title}</div><div class="tag-list">${tags}</div></div>`;
  };

  let ovHtml = `
    <h2>📊 ภาพรวมการตรวจ</h2>
    <div class="stat-grid">
      <div class="stat-box"><div class="label">ผลไม้ทั้งหมด</div><div class="value">${ov.total}</div></div>
      <div class="stat-box"><div class="label">ชนิดเกรด</div><div class="value">${Object.keys(ov.grades).length}</div></div>
      <div class="stat-box"><div class="label">ระดับสุก</div><div class="value">${Object.keys(ov.ripeness).length}</div></div>
      <div class="stat-box"><div class="label">ประเภทตำหนิ</div><div class="value">${Object.keys(ov.defects).length}</div></div>
    </div>
    <hr class="divider" />
    ${makeTagGroup('เกรด', ov.grades,'🍽️ ')}
    ${makeTagGroup('ความสุก', ov.ripeness,'🥭 ')}
    ${makeTagGroup('ตำหนิ / โรค', ov.defects,'⚠️ ')}
  `;
  overviewDiv.innerHTML = ovHtml;

  if(data.details && data.details.length){
    let dt = `<h2>รายละเอียดรายลูก</h2>`;
    dt += `<table border="1" style="margin:auto; border-collapse:collapse;">
      <tr><th>#</th><th>ID</th><th>Grade</th><th>Ripeness</th><th>Conf</th><th>Defects</th><th>ตำแหน่ง (box)</th></tr>`;
    data.details.forEach((d,i)=>{
      const defects = d.defects && d.defects.length ? d.defects.map(df=>`${df.name} (${df.confidence})`).join('<br>') : '-';
      dt += `<tr>
        <td>${i+1}</td>
        <td>${d.id ?? '-'}</td>
  <td>${d.grade ?? '-'}</td>
  <td>${d.ripeness ?? '-'}</td>
  <td>${d.ripeness_confidence ?? '-'}</td>
        <td>${defects}</td>
        <td>[${d.box.join(', ')}]</td>
      </tr>`;
    });
    dt += `</table>`;
    detailsDiv.innerHTML = dt;
  } else {
    detailsDiv.innerHTML = '';
  }
}

function pollLatest(){
  fetch('/latest_summary')
    .then(r=>r.json())
    .then(renderSummary)
    .catch(e=>console.warn('poll error',e));
}
if(userCam){
  // Deploy mode: capture user webcam and send frames to /predict
  async function initCam(){
    try{
      const stream = await navigator.mediaDevices.getUserMedia({video:true});
      userCam.srcObject = stream;
      if(statusEl) statusEl.textContent = 'Camera ready';
    }catch(e){
      if(statusEl) statusEl.textContent = 'Camera error: '+e.message;
    }
  }
  async function sendFrame(){
    if(userCam.readyState >= 2){
      const canvas = sendFrame._c || (sendFrame._c = document.createElement('canvas'));
      canvas.width = userCam.videoWidth;
      canvas.height = userCam.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(userCam,0,0);
      const dataURL = canvas.toDataURL('image/jpeg',0.7);
      try{
        const res = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:dataURL})});
        if(res.ok){
          const json = await res.json();
          renderSummary(json);
          if(statusEl) statusEl.textContent = 'Updated '+new Date().toLocaleTimeString();
        } else if(statusEl){ statusEl.textContent = 'Predict error '+res.status; }
      }catch(e){ if(statusEl) statusEl.textContent = 'Network error '+e.message; }
    }
    setTimeout(sendFrame, 1000);
  }
  initCam();
  sendFrame();
} else {
  // Local dev: poll summary
  setInterval(pollLatest,1500);
  pollLatest();
}
