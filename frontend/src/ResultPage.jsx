import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './ResultPage.css';

export default function ResultPage() {
  const navigate = useNavigate();
  
  const [activeTab, setActiveTab] = useState('summary');
  const [commentPlatform, setCommentPlatform] = useState('naver'); 

  const timelineData = [
    { id: 1, date: '2.01', volume: 15, phase: 'Phase 1', title: '정부, 의대 증원 발표', desc: '2025학년도 의대 정원 2000명 확대 방안 공식 발표' },
    { id: 2, date: '2.08', volume: 45, phase: '', title: '의료계 강경 대응 예고', desc: '의협 비대위 구성 및 총파업 등 집단 행동 논의 시작' },
    { id: 3, date: '2.15', volume: 100, phase: 'Phase 2', title: '전공의 집단 사직 시작', desc: '빅5 병원 전공의 사직서 제출, 의료 공백 현실화' },
    { id: 4, date: '2.20', volume: 85, phase: '', title: '정부 업무개시명령', desc: '복지부, 이탈 전공의 대상 면허 정지 경고 및 강경 대응' },
    { id: 5, date: '2.25', volume: 60, phase: 'Phase 3', title: '의정 갈등 장기화', desc: '타협점 없는 평행선, 환자 피해 속출 및 현장 혼란 가중' },
  ];
  const [activeTimeline, setActiveTimeline] = useState(timelineData[2]);

  const featureData = [
    { id: 'f1', label: '분노 (감정)', type: 'primary', top: '30%', left: '30%' },
    { id: 'f2', label: '전문가 견해 (논조)', type: 'primary', top: '70%', left: '70%' },
    { id: 'k1', parent: 'f1', label: '의료 붕괴', type: 'secondary', top: '15%', left: '20%', press: 'F뉴스', title: '응급실 뺑뺑이, 의료 붕괴 시작됐나', desc: '전공의 이탈로 인한 응급 환자 수용 거부 사태 집중 조명' },
    { id: 'k2', parent: 'f1', label: '무책임', type: 'secondary', top: '15%', left: '40%', press: 'A일보', title: '환자 볼모로 잡은 무책임한 파업', desc: '의사 본분을 저버린 집단행동에 대한 강도 높은 비판' },
    { id: 'k3', parent: 'f1', label: '졸속 행정', type: 'secondary', top: '45%', left: '15%', press: 'D저널', title: '준비 안 된 2000명, 전형적인 졸속', desc: '교육 인프라 확충 없는 일방적 증원 통보 지적' },
    { id: 'k4', parent: 'f2', label: '수도권 쏠림', type: 'secondary', top: '55%', left: '85%', press: 'E매체', title: '증원보다 시급한 건 지역 의료 살리기', desc: '단순 숫자 늘리기보단 지방 의료 유인책이 먼저라는 전문가 분석' },
    { id: 'k5', parent: 'f2', label: '필수의료', type: 'secondary', top: '85%', left: '60%', press: 'H신문', title: '기피 과목 수가 인상 없인 밑빠진 독', desc: '소아과, 산부인과 등 필수의료 붕괴 원인 분석 및 대안 제시' },
  ];
  const [activeFeature, setActiveFeature] = useState(featureData.find(f => f.id === 'k1'));

  const [biasX, setBiasX] = useState(50);
  const [biasY, setBiasY] = useState(50);

  const handleBiasSearch = () => {
    alert(`[백엔드 API 요청]\n진보/보수(X): ${biasX}\n분석/감정(Y): ${biasY}`);
  };

  return (
    <div className="whatsurf-container editorial-result">
      <div className="mesh-gradient-bg"></div>

      <header className="header">
        <div className="logo" onClick={() => navigate('/')}>whatsurf.</div>
        <button className="new-search-btn" onClick={() => navigate('/')}>New Search ↗</button>
      </header>

      <main className="result-main">
        <section className="result-hero">
          <h1 className="mega-title">의대 증원 논란</h1>
          <p className="result-meta">• 2,410개의 기사를 토대로 작성되었습니다.</p>
        </section>

        <nav className="result-tabs">
          <button className={`tab-btn ${activeTab === 'summary' ? 'active' : ''}`} onClick={() => setActiveTab('summary')}>기사 요약</button>
          <button className={`tab-btn ${activeTab === 'timeline' ? 'active' : ''}`} onClick={() => setActiveTab('timeline')}>타임 라인</button>
          <button className={`tab-btn ${activeTab === 'feature' ? 'active' : ''}`} onClick={() => setActiveTab('feature')}>피처맵</button>
          <button className={`tab-btn ${activeTab === 'map' ? 'active' : ''}`} onClick={() => setActiveTab('map')}>좌표 평면</button>
        </nav>

        <section className="tab-content-area">
          
          {/* ◼️ 탭 1: 기사 요약 */}
          {activeTab === 'summary' && (
            <div className="tab-summary fade-in">
              <div className="summary-cards-container">
                <div className="editorial-summary-card">
                  <div className="esc-left">
                    <div className="esc-header">
                      <span className="esc-badge primary">Mainstream</span>
                      <h3 className="esc-title">"의료 개혁은 필수불가결, 집단행동은 명분 없다"</h3>
                    </div>
                    <div className="esc-body">
                      <p className="esc-paragraph">
                        <span className="drop-cap">정</span>부의 의대 증원 정책을 강력히 지지하며, 의료계의 파업을 '국민 생명을 담보로 한 기득권 지키기'로 규정하는 논조가 주를 이룹니다.
                      </p>
                    </div>
                    <div className="esc-footer">
                      <span className="esc-press-label">Leading Press</span>
                      <div className="esc-press-logos">B신문, C방송, F뉴스 등 1,420건</div>
                    </div>
                  </div>

                  <div className="esc-right">
                    <div className="sns-filter-tabs">
                      <button className={commentPlatform === 'naver' ? 'active' : ''} onClick={() => setCommentPlatform('naver')}>Naver</button>
                      <button className={commentPlatform === 'insta' ? 'active' : ''} onClick={() => setCommentPlatform('insta')}>Instagram</button>
                      <button className={commentPlatform === 'other' ? 'active' : ''} onClick={() => setCommentPlatform('other')}>X (Twitter)</button>
                    </div>
                    <div className="sns-comments-feed">
                      <div className="sns-comment">
                        <div className="sns-user-info"><div className="avatar"></div><span>user_88***</span></div>
                        <p className="sns-text">
                          {commentPlatform === 'naver' && "환자 목숨 담보로 뭐하는 짓이냐 진짜 화난다. 기득권 지키기 역겹다."}
                          {commentPlatform === 'insta' && "🔥 파업 반대! 당장 다음 주 우리 엄마 수술 밀림 ㅠㅠ"}
                          {commentPlatform === 'other' && "이게 맞나 싶음... 당장 아픈 사람들은 어떡하라고 양쪽 다 노답인 듯."}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ◼️ 탭 2: 타임 라인 */}
          {activeTab === 'timeline' && (
            <div className="tab-timeline fade-in">
              <div className="editorial-timeline-card">
                <div className="massive-bg-date">{activeTimeline.date.replace('.', ' / ')}</div>
                <div className="strict-grid-layout">
                  <div className="editorial-chart-area">
                    <h3 className="editorial-label">TIMELINE VOLUME</h3>
                    <div className="sleek-bar-chart">
                      {timelineData.map((item) => (
                        <div key={item.id} className={`s-bar-wrap ${activeTimeline.id === item.id ? 'active' : ''}`} onClick={() => setActiveTimeline(item)}>
                          <div className="s-bar-track"><div className="s-bar-fill" style={{ height: `${item.volume}%` }}></div></div>
                          <span className="s-bar-date">{item.date}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="editorial-detail-area">
                    <div className="glass-panel detail-info-card">
                      <div className="detail-header">
                        {activeTimeline.phase && <span className="p-phase-badge">{activeTimeline.phase}</span>}
                        <span className="p-date-badge">{activeTimeline.date}</span>
                      </div>
                      <h3 className="d-article-title">"{activeTimeline.title}"</h3>
                      <p className="d-press-desc"><span className="drop-cap">보</span>{activeTimeline.desc}</p>
                      <button className="editorial-read-btn"><span>해당 시점 기사 보기</span> <span className="arrow">→</span></button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ◼️ 탭 3: 피처맵 (깨짐 현상 해결) */}
          {activeTab === 'feature' && (
            <div className="tab-feature fade-in">
              <div className="strict-grid-layout">
                <div className="glass-card feature-map-bg">
                  <h3 className="editorial-label" style={{position:'absolute', top:'30px', left:'30px', zIndex: 100}}>KEYWORD CLUSTERING</h3>
                  {featureData.map((node) => (
                    <div key={node.id} className={`f-node ${node.type} ${activeFeature?.id === node.id ? 'active' : ''}`} style={{ top: node.top, left: node.left }} onClick={() => node.type === 'secondary' && setActiveFeature(node)}>
                      {node.type === 'primary' && <span className="f-tag">핵심 피처</span>}
                      {node.label}
                    </div>
                  ))}
                </div>
                <div className="glass-card insight-panel">
                  <span className="insight-badge">FEATURE ARTICLE</span>
                  <div className="article-meta"><span className="meta-tag">연관 키워드: #{activeFeature.label}</span></div>
                  <h2 className="press-name">{activeFeature.press}</h2>
                  <h3 className="article-title">"{activeFeature.title}"</h3>
                  <p className="press-desc">{activeFeature.desc}</p>
                  <button className="editorial-read-btn"><span>기사 원문 보기</span> <span className="arrow">→</span></button>
                </div>
              </div>
            </div>
          )}

          {/* ◼️ 탭 4: 좌표 평면 (분포도 + 우측 짤림 해결) */}
          {activeTab === 'map' && (
            <div className="tab-map fade-in">
              <div className="strict-grid-layout">
                
                {/* 1) 좌측: 언론사 분포도 시각화 */}
                <div className="glass-card map-visual-container">
                  <h3 className="editorial-label" style={{marginBottom: '20px'}}>MEDIA DISTRIBUTION</h3>
                  
                  <div className="scatter-plot-area">
                    <div className="plot-grid"></div>
                    <div className="plot-axis x-axis"></div>
                    <div className="plot-axis y-axis"></div>
                    
                    <span className="axis-label x-left">진보</span>
                    <span className="axis-label x-right">보수</span>
                    <span className="axis-label y-top">분석적</span>
                    <span className="axis-label y-bottom">감정적</span>

                    {/* ✅ 점 대신 몽환적인 분포도(Blob) 영역으로 변경 */}
                    <div className="dist-blob" style={{ left: '25%', top: '75%', width: '180px', height: '140px', background: '#3b82f6' }}></div>
                    <div className="dist-blob" style={{ left: '75%', top: '35%', width: '220px', height: '160px', background: '#ec4899' }}></div>
                    <div className="dist-blob" style={{ left: '50%', top: '80%', width: '150px', height: '120px', background: '#f59e0b' }}></div>

                    {/* 분포도 라벨 */}
                    <span className="dist-text" style={{ left: '25%', top: '75%' }}>A일보 / D저널</span>
                    <span className="dist-text" style={{ left: '75%', top: '35%' }}>B신문 / C방송</span>
                    <span className="dist-text" style={{ left: '50%', top: '80%' }}>F뉴스</span>

                    {/* 타겟 십자선 */}
                    <div className="target-crosshair" style={{ left: `${biasX}%`, top: `${biasY}%` }}>
                      <div className="crosshair-center"></div>
                    </div>
                  </div>
                </div>

                {/* 2) 우측: 슬라이더 컨트롤러 + 매칭된 기사 패널 */}
                <div className="right-sidebar">
                  <div className="glass-card slider-control-panel">
                    <h3 className="control-title">TARGET SETTING</h3>
                    <p className="control-desc">원하는 기사의 성향과 논조를 조절하세요.</p>
                    
                    <div className="aesthetic-slider-group">
                      <div className="slider-labels"><span className="label-text">진보</span><span className="slider-value-pill">X : {biasX}</span><span className="label-text">보수</span></div>
                      <input type="range" min="0" max="100" value={biasX} onChange={(e) => setBiasX(e.target.value)} className="aesthetic-slider" style={{ '--val': `${biasX}%`, '--color': '#3b82f6' }} />
                    </div>

                    <div className="aesthetic-slider-group">
                      <div className="slider-labels"><span className="label-text">분석적</span><span className="slider-value-pill">Y : {biasY}</span><span className="label-text">감정적</span></div>
                      <input type="range" min="0" max="100" value={biasY} onChange={(e) => setBiasY(e.target.value)} className="aesthetic-slider" style={{ '--val': `${biasY}%`, '--color': '#ec4899' }} />
                    </div>
                  </div>

                  <div className="glass-card insight-panel map-insight">
                    <span className="insight-badge">MATCHED ARTICLE</span>
                    <h2 className="press-name">B신문</h2>
                    <h3 className="article-title">"의료계 파업, 결국 환자만 볼모 잡혔다"</h3>
                    <p className="press-desc">설정하신 성향(X:{biasX}, Y:{biasY})에 가장 가까운 bias_vector 기사입니다.</p>
                    {/* ✅ 디자인 업그레이드 된 버튼 */}
                    <button className="editorial-read-btn"><span>기사 원문 읽기</span> <span className="arrow">→</span></button>
                  </div>
                </div>

              </div>
            </div>
          )}

        </section>
      </main>
    </div>
  );
}