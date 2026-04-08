import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './MainPage.css';

const initialGameDataSet = [
  { id: 1, name: '피자', volume: 21100000, category: 'Food', img: 'https://images.unsplash.com/photo-1513104890138-7c749659a591?w=800' },
  { id: 2, name: '껌', volume: 1540000, category: 'Snack', img: 'https://images.unsplash.com/photo-1621285093557-08ce9f8dc979?w=800' },
  { id: 3, name: '아이폰 16', volume: 8900000, category: 'Tech', img: 'https://images.unsplash.com/photo-1664478546384-d2bfe603d3c2?w=800' },
  { id: 4, name: '유튜브', volume: 45000000, category: 'Web', img: 'https://images.unsplash.com/photo-1611162616305-c67b3fa40904?w=800' },
  { id: 5, name: '러닝화', volume: 3200000, category: 'Fashion', img: 'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=800' },
  { id: 6, name: '스타벅스', volume: 12500000, category: 'Cafe', img: 'https://images.unsplash.com/photo-1596716075908-163e77a28ebf?w=800' },
  { id: 7, name: '넷플릭스', volume: 28900000, category: 'OTT', img: 'https://images.unsplash.com/photo-1574375927938-d5a98e8ffe85?w=800' },
  { id: 8, name: '비트코인', volume: 68000000, category: 'Finance', img: 'https://images.unsplash.com/photo-1518546305927-5a555bb7020d?w=800' },
  { id: 9, name: '한강 공원', volume: 550000, category: 'Place', img: 'https://images.unsplash.com/photo-1605371661332-6a7590d93026?w=800' },
  { id: 10, name: 'GPT-4', volume: 18700000, category: 'AI', img: 'https://images.unsplash.com/photo-1678911820864-a2c96e334111?w=800' },
];

export default function MainPage() {
  const [keyword, setKeyword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const [gameData, setGameData] = useState(initialGameDataSet);
  const [leftItem, setLeftItem] = useState(initialGameDataSet[0]);
  const [rightItem, setRightItem] = useState(initialGameDataSet[1]);
  const [currentIndex, setCurrentIndex] = useState(1);
  const [currentScore, setCurrentScore] = useState(0);
  const [highScore, setHighScore] = useState(0);
  const [rightVolumeVisible, setRightVolumeVisible] = useState(false);

  // ✅ 1. 검색어 오류 처리 (빈 칸, 글자 수 제한 시 안 넘어가게 수정)
  const handleSearch = (e) => {
    if (e) e.preventDefault();
    const trimmedKeyword = keyword.trim();
    
    if (trimmedKeyword.length === 0) { alert('ERROR!\n검색어를 입력해주세요!'); return; }
    if (trimmedKeyword.length > 15) { alert('ERROR!\n검색어가 제한 된 글자 수를 초과하였습니다.'); return; }
    if (trimmedKeyword.length < 2) { alert('ERROR!\n정확한 검색어(2자 이상)를 입력해주세요!'); return; }
    
    setIsLoading(true);
    setTimeout(() => { navigate('/result', { state: { searchTerm: trimmedKeyword } }); }, 1500);
  };

  // ✅ 2. 실시간 트렌드 클릭 시 검색 실행
  const handleTrendClick = (trendKeyword) => {
    setKeyword(trendKeyword);
    setIsLoading(true);
    setTimeout(() => { navigate('/result', { state: { searchTerm: trendKeyword } }); }, 1500);
  };

  const chooseHigher = () => { rightItem.volume >= leftItem.volume ? handleCorrect() : handleWrong(); };
  const chooseLower = () => { rightItem.volume <= leftItem.volume ? handleCorrect() : handleWrong(); };

  const handleCorrect = () => {
    setRightVolumeVisible(true);
    setTimeout(() => {
      const nextScore = currentScore + 1;
      setCurrentScore(nextScore);
      if (nextScore > highScore) setHighScore(nextScore);
      setLeftItem(rightItem);
      const nextIndex = currentIndex + 1;
      if (nextIndex >= gameData.length) {
        resetGame("Congratulation! 데이터를 다 클리어하셨습니다.");
      } else {
        setRightItem(gameData[nextIndex]);
        setCurrentIndex(nextIndex);
        setRightVolumeVisible(false);
      }
    }, 800);
  };

  const handleWrong = () => {
    alert(`오답입니다!\n${rightItem.name}의 검색량은 ${rightItem.volume.toLocaleString()}회 였습니다.\n게임이 초기화됩니다.`);
    resetGame();
  };

  const resetGame = (msg) => {
    if(msg) alert(msg);
    setCurrentScore(0);
    setCurrentIndex(1);
    setLeftItem(initialGameDataSet[0]);
    setRightItem(initialGameDataSet[1]);
    setRightVolumeVisible(false);
  };

  return (
    <div className="whatsurf-container editorial-page">
      <div className="mesh-gradient-bg"></div>

      <header className="header">
        <div className="logo">whatsurf.</div>
        <div className="trending-area">
          <span className="trend-title">Hot Issue</span>
          {/* ✅ 3. 트렌드 키워드에 클릭 이벤트 연결 */}
          <div className="trend-words">
            <span className="trend-clickable" onClick={() => handleTrendClick('의대 증원')}>의대 증원</span> &nbsp;·&nbsp; 
            <span className="trend-clickable" onClick={() => handleTrendClick('딥페이크')}>딥페이크</span> &nbsp;·&nbsp; 
            <span className="trend-clickable" onClick={() => handleTrendClick('금투세')}>금투세</span>
          </div>
        </div>
      </header>

      <main className="main-content">
        <section className="hero-section">
          <div className="hero-left">
            <h1 className="mega-text">READ<br/>THE<br/>UNSEEN.</h1>
          </div>
          
          <div className="hero-right">
            <p className="hero-desc">표면적인 기사를 넘어, 대중의 숨겨진<br/>진짜 프레임과 감정을 읽어냅니다.</p>
            <form className="search-form" onSubmit={handleSearch}>
              <div className={`search-input-wrapper ${isLoading ? 'loading' : ''}`}>
                <input type="text" placeholder="어떤 사건의 이면이 궁금하신가요?" value={keyword} onChange={(e) => setKeyword(e.target.value)} className="search-input" disabled={isLoading} />
                <button type="submit" className="search-btn" disabled={isLoading}>{isLoading ? '탐색 중...' : 'Discover ↗'}</button>
              </div>
              <div className="search-bottom-area">
                {isLoading ? (
                  <div className="elegant-loading"><div className="progress"></div></div>
                ) : (
                  <p className="search-tip">TIP. 주요 키워드로 숨겨진 프레임을 확인해 보세요.</p>
                )}
              </div>
            </form>
          </div>
        </section>

        <section className="game-section">
          <div className="game-header">
            <span className="game-title-label">Search Volume Battle</span>
            <div className="score-area">
              <div className="score">Best <span>{highScore}</span></div>
              <div className="score">Now <span>{currentScore}</span></div>
            </div>
          </div>

          <div className="game-wrapper">
            <div className="game-panel left-panel" style={{ backgroundImage: `url(${leftItem.img})` }}>
              <div className="panel-overlay"></div>
              <div className="glass-card">
                <span className="category-tag">{leftItem.category}</span>
                <h2 className="game-keyword">{leftItem.name}</h2>
                <p className="search-volume">{leftItem.volume.toLocaleString()}회</p>
                <p className="vol-status">검색됨</p>
              </div>
            </div>

            <div className="vs-circle">VS</div>

            <div className="game-panel right-panel" style={{ backgroundImage: `url(${rightItem.img})` }}>
              <div className="panel-overlay"></div>
              <div className="glass-card">
                <span className="category-tag">{rightItem.category}</span>
                <h2 className="game-keyword">{rightItem.name}</h2>
                {rightVolumeVisible ? (
                  <p className="search-volume fade-in">{rightItem.volume.toLocaleString()}회</p>
                ) : (
                  <p className="search-volume hide-vol">?</p>
                )}
                <p className="vol-status">회 검색?</p>

                <div className="game-btns">
                  <button className="action-btn" onClick={chooseHigher}>더 많이 ▲</button>
                  <button className="action-btn outline" onClick={chooseLower}>더 적게 ▼</button>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <span>Instagram @surf.search</span>
        <span>unnamed@gmail.com</span>
      </footer>
    </div>
  );
}