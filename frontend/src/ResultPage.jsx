import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './ResultPage.css';

const SUPABASE_URL = 'http://localhost:8000';
const SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5yc2N4aG95ZGxuY2pxcHR2am1jIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3Mzg4MjAxMywiZXhwIjoyMDg5NDU4MDEzfQ.N_T0rYbxFb_qPdM4KPr7qZX2YjG0IhlDd2Q50RwrzYY';

// --- [강화된 더미데이터] ---
const DUMMY_CLUSTERS = [
  { 
    id: 1, cluster_label: 0, cluster_title: "정부-의료계 강대강 대치 심화", cluster_count: 1240, 
    cluster_summary: "정부의 의대 정원 2000명 증원 확정 발표 이후, 전공의들의 집단 사직과 의대생들의 휴학 신청이 이어지며 의료 공백이 현실화되고 있습니다.", 
    leading_sources: [{source_name: '중앙일보'}, {source_name: '연합뉴스'}, {source_name: 'KBS'}], 
    representative_comments: [
      {comment_id: 1, cmt_content: "환자들만 볼모로 잡는 처사입니다. 당장 복귀하세요.", cmt_emotion: "분노"},
      {comment_id: 2, cmt_content: "교육 여건 부재를 주장하는 의료계 입장도 이해갑니다.", cmt_emotion: "중립"}
    ] 
  },
  { 
    id: 2, cluster_label: 1, cluster_title: "응급실 운영난 및 시민 불안 확산", cluster_count: 850, 
    cluster_summary: "전공의 이탈이 장기화되면서 주요 상급종합병원의 응급실이 파행 운영되고 있습니다. 이른바 '응급실 뺑뺑이' 사례가 속출하며 시민들의 불안감이 커지고 있습니다.", 
    leading_sources: [{source_name: 'SBS'}, {source_name: '한겨레'}, {source_name: 'MBC'}], 
    representative_comments: [
      {comment_id: 3, cmt_content: "아이가 아플 때 갈 곳이 없을까봐 너무 불안해요.", cmt_emotion: "불안"}
    ] 
  }
];

const DUMMY_FEATURE_MAPS = {
  cluster_0: [
    { id: 'p1', label: '사건 원인 집중', type: 'primary', top: '35%', left: '30%', color: '#3b82f6', desc: '의료 인력 부족 현상 분석' },
    { id: 's1', label: '필수의료 붕괴', type: 'secondary', top: '15%', left: '20%', ratio: 0.8 },
  ]
};

const DUMMY_MEDIA_DIST = [
  { id: 1, source_name: '조선일보', center_x: 0.85, center_y: 0.6, spread_radius: 0.15 },
  { id: 2, source_name: '한겨레', center_x: -0.8, center_y: 0.7, spread_radius: 0.2 },
];

export default function ResultPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const searchTerm = location.state?.searchTerm || "의대 증원 논란";

  // 상태 관리 (타임라인 탭이 돌아왔습니다!)
  const [activeTab, setActiveTab] = useState('summary');
  const [mapSubTab, setMapSubTab] = useState('2d');
  const [selectedCluster, setSelectedCluster] = useState(0); 
  const [activeFeature, setActiveFeature] = useState(null);
  const [commentPlatform, setCommentPlatform] = useState('naver'); 
  const [isLoading, setIsLoading] = useState(true);

  // 데이터 상태
  const [queryId, setQueryId] = useState(null);
  const [totalArticles, setTotalArticles] = useState(0);
  const [clusterDetails, setClusterDetails] = useState([]);
  const [timelineData, setTimelineData] = useState([]);
  const [activeTimeline, setActiveTimeline] = useState(null);
  const [featureMapsData, setFeatureMapsData] = useState(null); 
  const [distributionData, setDistributionData] = useState([]);
  const [biasX, setBiasX] = useState(50);
  const [biasY, setBiasY] = useState(50);
  const [matchedArticle, setMatchedArticle] = useState(null);
  const [expandedGroup, setExpandedGroup] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [statusMessage, setStatusMessage] = useState('서버 분석 요청 중...');

  const currentNodes = featureMapsData ? (featureMapsData[selectedCluster] || []) : (DUMMY_FEATURE_MAPS[`cluster_${selectedCluster}`] || []);

  const fetchAPI = async (endpoint, method = 'GET', body = null) => {
    try {
      const options = {
        method, headers: { 'Content-Type': 'application/json', 'apikey': SUPABASE_KEY, 'Authorization': `Bearer ${SUPABASE_KEY}` }
      };
      if (body) options.body = JSON.stringify(body);
      const response = await fetch(`${SUPABASE_URL}${endpoint}`, options);
      if (!response.ok) throw new Error("Server Offline");
      return await response.json();
    } catch (err) {
      return null; 
    }
  };

  useEffect(() => {
    let isMounted = true;

    const fetchAllData = async () => {
      try {
        setIsLoading(true);
        console.log(`🔍 [검색 시작] "${searchTerm}" 서버 요청`);

        const postData = await fetchAPI('/api/queries', 'POST', { query_text: searchTerm });
        const currentQueryId = postData?.data?.id || postData?.id || "dummy-id";
        const isCached = postData?.data?.cached;
        if (isMounted) setQueryId(currentQueryId);

        let clusters = [];

        if (isCached) {
          // 캐시 있음 → 즉시 조회
          setStatusMessage('캐시된 분석 결과를 불러오는 중...');
          const clusterData = await fetchAPI(`/api/queries/${currentQueryId}/clusters`);
          clusters = clusterData?.data?.clusters || clusterData?.clusters || [];
        } else {
          // 캐시 없음 → 새로 분석 중, 최대 60분 폴링
          setStatusMessage('새로운 키워드를 분석 중입니다. 크롤링과 AI 분석에 최대 60분 정도 걸릴 수 있어요...');
          const maxTries = 360; // 360회 * 10초 = 60분
          for (let i = 1; i <= maxTries; i++) {
            if (!isMounted) return;
            const clusterData = await fetchAPI(`/api/queries/${currentQueryId}/clusters`);
            clusters = clusterData?.data?.clusters || clusterData?.clusters || [];
            if (clusters.length > 0) break;

            // 경과 시간 표시 (선택)
            const elapsedMin = Math.floor((i * 10) / 60);
            setStatusMessage(`새로운 키워드를 분석 중입니다... (경과: 약 ${elapsedMin}분)`);

            if (i < maxTries) await new Promise(r => setTimeout(r, 10000));
          }
        }

        if (!isMounted) return;

        if (clusters.length > 0) {
          const detailedClusters = await Promise.all(
            clusters.map(async (c) => {
              const detail = await fetchAPI(`/api/queries/${currentQueryId}/clusters/${c.cluster_label}`);
              return { ...c, ...detail?.data };
            })
          );
          setClusterDetails(detailedClusters);
          setTotalArticles(detailedClusters.reduce((sum, c) => sum + (Number(c.article_count) || 0), 0));
        } else {
          console.warn("분석 실패 또는 타임아웃(60분) → 더미 표시");
          setClusterDetails(DUMMY_CLUSTERS);
          setTotalArticles(3240);
          setIsLoading(false);
          return;
        }

        if (!isMounted) return;

        if (clusters.length > 0) {
          const detailedClusters = await Promise.all(
            clusters.map(async (c) => {
              const detail = await fetchAPI(`/api/queries/${currentQueryId}/clusters/${c.cluster_label}`);
              return { ...c, ...detail?.data };
            })
          );
          setClusterDetails(detailedClusters);
          setTotalArticles(detailedClusters.reduce((sum, c) => sum + (Number(c.article_count) || 0), 0));
        } else {
          console.warn("분석 실패 또는 타임아웃 → 더미 표시");
          setClusterDetails(DUMMY_CLUSTERS);
          setTotalArticles(3240);
          setIsLoading(false);
          return;
        }
        // [추가된 타임라인 로직]
        try {
          const timelineJson = await fetchAPI(`/api/queries/${currentQueryId}/timeline`);
          const timelines = timelineJson?.data?.timelines || timelineJson?.timelines || [];
          if (timelines.length > 0) {
            const formattedTimeline = timelines.map((item, index) => {
              const d = new Date(item.timeline_date || Date.now());
              return {
                id: item.id || index,
                date: `${d.getMonth() + 1}.${String(d.getDate()).padStart(2, '0')}`,
                title: item.article_title, url: item.article_url, volume: 40 + Math.random() * 40,
                desc: item.summary || "", phase: index === 0 ? "Initial" : ""
              };
            });
            setTimelineData(formattedTimeline);
            setActiveTimeline(formattedTimeline[0]);
          } else {
             const dummyTimeline = [
              { id: 1, date: '02.06', volume: 20, phase: 'Phase 1', title: '의대 증원 발표', desc: '정부 공식 브리핑.', url: '#' },
              { id: 2, date: '02.20', volume: 100, phase: 'Phase 2', title: '전공의 사직', desc: '의료 대란 본격화.', url: '#' }
            ];
            setTimelineData(dummyTimeline);
            setActiveTimeline(dummyTimeline[1]);
          }
        } catch (e) { console.warn("타임라인 로드 실패", e); }


        try {
          const newFeatureMaps = {};
          for (const c of clusters) {
            const fmRes = await fetchAPI(`/api/queries/${currentQueryId}/clusters/${c.cluster_label}/feature-map`);
            const fm = fmRes?.data?.feature_map || fmRes?.feature_map;
            
            if (fm) {
              const nodes = [];
              const threshold = 0.3; 

              nodes.push({ id: `p_frame_${c.cluster_label}`, label: '보도 프레임', type: 'primary', top: '20%', left: '30%', color: '#3b82f6', desc: '기사의 주요 프레임' });
              nodes.push({ id: `p_logic_${c.cluster_label}`, label: '논조', type: 'primary', top: '70%', left: '60%', color: '#10b981', desc: '기사의 논리적 스탠스' });
              nodes.push({ id: `p_emo_${c.cluster_label}`, label: '감정선', type: 'primary', top: '40%', left: '80%', color: '#ec4899', desc: '내포된 주요 감정' });

              const addSubNodes = (items, parentId, startTop, startLeft, color) => {
                if (!items) return;
                items.filter(item => item.ratio > threshold).forEach((item, idx) => {
                  const words = item.loaded_words ? item.loaded_words.join(', ') : (item.stance_score ? `${item.stance_score.label} (${item.stance_score.value})` : '');
                  nodes.push({
                    id: `${parentId}_sub_${idx}`,
                    label: item.label,
                    type: 'secondary',
                    ratio: item.ratio, 
                    top: `${startTop + (idx * 15)}%`,
                    left: `${startLeft + (idx * 12)}%`,
                    color: color,
                    desc: `[비중: ${Math.round(item.ratio * 100)}%] \n${words ? `주요 단어: ${words}` : ''}`
                  });
                });
              };

              addSubNodes(fm.frame, `p_frame_${c.cluster_label}`, 5, 10, '#3b82f6');
              addSubNodes(fm.logic, `p_logic_${c.cluster_label}`, 55, 45, '#10b981');
              addSubNodes(fm.emotions, `p_emo_${c.cluster_label}`, 25, 75, '#ec4899');

              newFeatureMaps[c.cluster_label] = nodes;
            }
          }
          if (Object.keys(newFeatureMaps).length > 0) setFeatureMapsData(newFeatureMaps);
        } catch (e) { console.warn("피처맵 로드 실패", e); }

        try {
          const biasJson = await fetchAPI(`/api/queries/${currentQueryId}/bias-plane`);
          const mediaDist = biasJson?.data?.media_distribution || biasJson?.media_distribution || [];
          const colors = ['#ec4899', '#3b82f6', '#f59e0b', '#10b981', '#8b5cf6'];
          
          if (mediaDist.length > 0) {
            // center_x 범위 파악해서 전체 화면에 퍼뜨리기
            const xs = mediaDist.map(d => d.center_x);
            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const rangeX = maxX - minX || 1;

            const ys = mediaDist.map(d => d.center_y);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);
            const rangeY = maxY - minY || 1;

            const mappedDist = mediaDist.map((item, idx) => {
              // 10%~90% 범위로 정규화해서 퍼뜨림
              const leftPos = 10 + ((item.center_x - minX) / rangeX) * 80;
              const topPos = 90 - ((item.center_y - minY) / rangeY) * 80;
              const bubbleSize = 60 + ((item.article_count || 1) * 2);

              return {
                ...item,
                left: leftPos,
                top: topPos,
                size: bubbleSize,
                color: colors[idx % colors.length]
              };
            });
            setDistributionData(mappedDist);
            console.log('distributionData:', distributionData.map(d => ({ name: d.source_name, left: d.left })));
          }
        } catch (e) { console.warn("좌표평면 로드 실패", e); }

      } catch (error) {
        console.error("데이터 세팅 중 오류:", error);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };

    fetchAllData();
    return () => { isMounted = false; };
  }, [searchTerm]);

    const handleBiasSearch = async () => {
    // 슬라이더 0~100 → API용 -1~1 변환
    const queryX = (biasX / 50) - 1;

    try {
      const matchData = await fetchAPI(`/api/queries/${queryId}/bias-plane/match?x=${queryX}&y=0`);
      if (matchData?.data?.matched_article) {
        const article = matchData.data.matched_article;
        setMatchedArticle({
          title: article.title,
          url: article.url
        });
      }
    } catch (error) {
      console.error("매칭 기사 검색 실패:", error);
    }
  };

  return (
    <div className="whatsurf-container editorial-result">
      <div className="mesh-gradient-bg"></div>

      <header className="header">
        <div className="logo" onClick={() => navigate('/')} style={{cursor:'pointer'}}>whatsurf.</div>
        <button className="new-search-btn" onClick={() => navigate('/')}>New Search ↗</button>
      </header>

      <main className="result-main">
        <section className="result-hero">
          <h1 className="mega-title">{searchTerm}</h1>
          <p className="result-meta">• {isLoading ? '서버 분석 요청 중...' : `${(totalArticles||0).toLocaleString()}개의 기사 기반 분석 완료`}</p>
        </section>

        <nav className="result-tabs">
          <button className={`tab-btn ${activeTab === 'summary' ? 'active' : ''}`} onClick={() => setActiveTab('summary')}>기사 요약</button>
          <button className={`tab-btn ${activeTab === 'timeline' ? 'active' : ''}`} onClick={() => setActiveTab('timeline')}>타임 라인</button>
          <button className={`tab-btn ${activeTab === 'feature' ? 'active' : ''}`} onClick={() => setActiveTab('feature')}>피처맵</button>
          <button className={`tab-btn ${activeTab === 'map' ? 'active' : ''}`} onClick={() => setActiveTab('map')}>좌표 평면</button>
        </nav>

        <section className="tab-content-area">
          {isLoading ? (
            <div className="loading-container" style={{ textAlign: 'center', padding: '100px 0' }}>
              <div className="spinner"></div>
              <p style={{ marginTop: '20px', color: '#666' }}>데이터를 수집하고 분석 중입니다... (10초 주기)</p>
            </div>
          ) : (
            <>
              {/* ◼️ 탭 1: 기사 요약 */}
              {activeTab === 'summary' && (
                <div className="tab-summary fade-in">
                  <div className="summary-cards-container">
                    {clusterDetails.map((cluster, index) => (
                      <div className="editorial-summary-card" key={index}>
                        <div className="esc-left">
                          <div className="esc-header">
                            <span className={`esc-badge ${index === 0 ? 'primary' : 'secondary'}`}>{index === 0 ? 'Mainstream' : 'Minority'}</span>
                            <h3 className="esc-title">"{cluster.cluster_title}"</h3>
                          </div>
                          <div className="esc-body">
                            <p className="esc-paragraph">
                              <span className="drop-cap">{cluster.cluster_summary?.charAt(0) || '분'}</span>
                              {cluster.cluster_summary?.slice(1) || '석된 요약이 없습니다.'}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ◼️ 탭 2: 타임라인 */}
              {activeTab === 'timeline' && activeTimeline && (
                <div className="tab-timeline fade-in">
                  <div className="glass-card editorial-timeline-card">
                    <div className="massive-bg-date">{activeTimeline.date.replace('.', ' / ')}</div>
                    <div className="timeline-grid-layout">
                      <div className="timeline-chart-area">
                        <div className="volume-bar-container">
                          {timelineData.map((item) => (
                            <div key={item.id} className={`v-bar-group ${activeTimeline.id === item.id ? 'active' : ''}`} onClick={() => setActiveTimeline(item)}>
                              <div className="v-bar-track"><div className="v-bar-fill" style={{ height: `${item.volume}%` }}></div></div>
                              <span className="v-bar-date">{item.date}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="timeline-detail-area">
                        <div className="timeline-info-box">
                          <div className="detail-header"><span className="p-phase-badge">{activeTimeline.phase}</span><span className="p-date-badge">{activeTimeline.date}</span></div>
                          <div className="timeline-text-content" style={{marginTop:'20px'}}>
                            <h3 className="d-article-title">{activeTimeline.title}</h3>
                            <p className="d-press-desc">{activeTimeline.desc}</p>
                          </div>
                          <button className="editorial-read-btn glass-btn" onClick={() => window.open(activeTimeline.url, '_blank')}>원문 읽기 →</button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* ◼️ 탭 3: 피처맵 */}
              {activeTab === 'feature' && (
                <div className="tab-feature fade-in">
                  <div className="map-sub-nav" style={{display:'flex', gap:'10px', marginBottom:'20px'}}>
                    {clusterDetails.map((c, i) => (
                      <button key={i} className={`sub-tab-btn ${selectedCluster === c.cluster_label ? 'active' : ''}`} onClick={() => setSelectedCluster(c.cluster_label)}>관점 {i+1} 분석</button>
                    ))}
                  </div>

                  <div className="strict-grid-layout">
                    <div className="glass-card feature-map-bg" style={{minHeight:'600px', position:'relative', overflow:'hidden'}}>
                      {currentNodes.length > 0 ? currentNodes.map((node) => {
                        const dynamicSize = node.type === 'primary' ? '130px' : `${(node.ratio * 200) + 50}px`;
                        
                        return (
                          <div 
                            key={node.id} 
                            className={`f-node ${node.type} ${activeFeature?.id === node.id ? 'active' : ''}`} 
                            style={{ 
                              top: node.top, 
                              left: node.left, 
                              width: dynamicSize,
                              height: dynamicSize,
                              backgroundColor: node.type === 'primary' ? '#111' : 'rgba(255,255,255,0.95)',
                              border: node.type === 'primary' ? `3px solid ${node.color}` : `2px solid ${node.color}`,
                              color: node.type === 'primary' ? '#fff' : '#111',
                              boxShadow: activeFeature?.id === node.id ? `0 0 20px ${node.color}55` : 'none'
                            }} 
                            onClick={() => setActiveFeature(node)}
                          >
                            <span style={{fontWeight:'700', textAlign:'center', fontSize: node.type === 'primary' ? '1rem' : '0.85rem'}}>{node.label}</span>
                          </div>
                        );
                      }) : <p style={{padding:'20px'}}>선택된 군집의 피처맵 데이터가 없습니다.</p>}
                    </div>

                    <div className="glass-card insight-panel">
                      <span className="insight-badge">FEATURE DETAIL</span>
                      {activeFeature ? (
                        <>
                          <h2 className="press-name" style={{marginTop:'20px', color: activeFeature.color}}>{activeFeature.type === 'primary' ? '분석 카테고리' : '상세 키워드'}</h2>
                          <h3 className="article-title" style={{fontSize:'1.4rem'}}>{activeFeature.label}</h3>
                          <p className="press-desc" style={{marginTop:'15px', whiteSpace: 'pre-line', lineHeight: '1.6'}}>{activeFeature.desc}</p>
                        </>
                      ) : (
                        <p style={{marginTop:'40px', color:'#999'}}>버블을 클릭하시면 상세 분석 결과가 표시됩니다.</p>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* ◼️ 탭 4: 1D 선형 분포 */}
              {activeTab === 'map' && (
                <div className="tab-map fade-in">
                  <div className="glass-card" style={{ padding: '40px 50px' }}>
                    
                    {/* 진보 / 보수 레이블 */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <span style={{ fontSize: '0.85rem', color: '#3b82f6', fontWeight: '600' }}>← 진보</span>
                      <span style={{ fontSize: '0.85rem', color: '#ec4899', fontWeight: '600' }}>보수 →</span>
                    </div>

                    {/* 선 + 버블 영역 */}
                    <div
                      style={{ position: 'relative', height: '160px', margin: '0 0 20px 0', cursor: 'pointer' }}
                      onMouseDown={(e) => {
                        setIsDragging(true);
                        const rect = e.currentTarget.getBoundingClientRect();
                        const x = Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100));
                        setBiasX(x);
                      }}
                      onMouseMove={(e) => {
                        if (!isDragging) return;
                        const rect = e.currentTarget.getBoundingClientRect();
                        const x = Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100));
                        setBiasX(x);
                      }}
                      onMouseUp={() => setIsDragging(false)}
                      onMouseLeave={() => setIsDragging(false)}
                    >
                      {/* 수평선 */}
                      <div style={{
                        position: 'absolute', top: '80px', left: 0, right: 0,
                        height: '2px', background: '#ddd'
                      }} />

                      {/* 버블들 */}
                      {(() => {
                        const groups = [];
                        distributionData.forEach(media => {
                          const existing = groups.find(g => Math.abs(g.left - media.left) < 2);
                          if (existing) {
                            existing.items.push(media);
                          } else {
                            groups.push({ left: media.left, items: [media] });
                          }
                        });

                        return groups.map((group, idx) => {
                          const isTop = idx % 2 === 0;
                          const bubbleTop = isTop ? 40 : 120;
                          const isMultiple = group.items.length > 1;
                          const isExpanded = expandedGroup === idx;

                          return (
                            <div key={idx} style={{
                              position: 'absolute',
                              left: `${group.left}%`,
                              top: `${bubbleTop}px`,
                              transform: 'translate(-50%, -50%)',
                              zIndex: isExpanded ? 20 : 1,
                              cursor: isMultiple ? 'pointer' : 'default'
                            }}
                              onClick={() => isMultiple && setExpandedGroup(isExpanded ? null : idx)}
                            >
                              <div style={{
                                width: `${group.items[0].size}px`,
                                height: `${group.items[0].size}px`,
                                borderRadius: '50%',
                                background: group.items[0].color,
                                opacity: 0.55,
                                transform: 'translate(-50%, -50%)',
                                position: 'absolute', top: 0, left: 0,
                                border: isMultiple ? '2px dashed #666' : 'none'
                              }} />

                              <span style={{
                                position: 'absolute',
                                top: isTop
                                  ? `-${group.items[0].size / 2 + 18}px`
                                  : `${group.items[0].size / 2 + 6}px`,
                                left: '50%', transform: 'translateX(-50%)',
                                fontSize: '0.72rem', whiteSpace: 'nowrap',
                                color: isMultiple ? '#111' : '#444',
                                fontWeight: isMultiple ? '700' : '500'
                              }}>
                                {isMultiple
                                  ? `${group.items[0].source_name} 외 ${group.items.length - 1}개`
                                  : group.items[0].source_name}
                              </span>

                              {isExpanded && (
                                <div style={{
                                  position: 'absolute',
                                  top: isTop
                                    ? `-${group.items[0].size / 2 + 20 + group.items.length * 22}px`
                                    : `${group.items[0].size / 2 + 10}px`,
                                  left: '50%', transform: 'translateX(-50%)',
                                  background: '#fff', border: '1px solid #ddd',
                                  borderRadius: '8px', padding: '8px 12px',
                                  boxShadow: '0 4px 12px rgba(0,0,0,0.12)',
                                  minWidth: '120px'
                                }}>
                                  {group.items.map((item, i) => (
                                    <div key={i} style={{
                                      fontSize: '0.75rem', padding: '3px 0',
                                      color: '#333', whiteSpace: 'nowrap',
                                      borderBottom: i < group.items.length - 1 ? '1px solid #f0f0f0' : 'none'
                                    }}>
                                      {item.source_name}
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          );
                        });
                      })()}

                      {/* 포인터 */}
                      <div style={{
                        position: 'absolute',
                        left: `${biasX}%`, top: '80px',
                        transform: 'translate(-50%, -50%)',
                        width: '18px', height: '18px',
                        borderRadius: '50%', background: '#111',
                        border: '3px solid #fff',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
                        zIndex: 10, pointerEvents: 'none'
                      }} />
                    </div>

                    {/* 기사 찾기 버튼 */}
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '24px' }}>
                      <button className="search-target-btn" onClick={handleBiasSearch}
                        style={{ padding: '14px 60px', fontSize: '1rem' }}>
                        기사 찾기
                      </button>
                    </div>

                    {/* 결과 */}
                    {matchedArticle ? (
                      <div style={{ textAlign: 'center', paddingTop: '16px', borderTop: '1px solid #eee' }}>
                        <span className="insight-badge">검색 완료</span>
                        <h3 className="article-title" style={{ wordBreak: 'keep-all', lineHeight: '1.5', margin: '16px 0' }}>
                          {matchedArticle.title}
                        </h3>
                        <button
                          className="editorial-read-btn glass-btn"
                          onClick={() => window.open(matchedArticle.url, '_blank')}
                        >
                          기사 원문 읽기 →
                        </button>
                      </div>
                    ) : (
                      <p style={{ color: '#999', textAlign: 'center', paddingTop: '16px', borderTop: '1px solid #eee' }}>
                        슬라이더를 조절해 원하는 성향의 기사를 찾아보세요.
                      </p>
                    )}

                  </div>
                </div>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}